import argparse
import asyncio
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure package root and local runner dir are on sys.path
_SERVER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SERVER_DIR.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from runner.ai_assistant import KaggleAIAssistant
from runner.job_builder import (
    OUTPUTS_DIR,
    WORK_DIR,
    download_job_output,
    kernel_id,
    prepare_project_jobs,
    push_job,
    run_parallel_tasks,
    status_job,
    stop_job,
)
import concurrent.futures
from runner.kaggle_api import (
    cancel_kernel,
    clear_api_cache,
    get_account_gpu_status,
    read_token,
    run_kaggle,
    validate_account,
    write_access_token,
    write_kaggle_json,
)
from runner.log_reader import (
    build_comparison_table,
    collect_finetune_status,
    combine_summaries,
    find_job_files,
    read_csv,
    read_jsonl,
    read_text_tail,
)
from runner.project_config import (
    ROOT,
    accounts_by_id,
    add_account,
    add_job_to_project,
    delete_account,
    delete_job_from_project,
    load_accounts,
    load_projects,
    resolve_tool_path,
    save_accounts,
    save_projects,
)

app = FastAPI(title="Kaggle Multi-Account Hub API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant = KaggleAIAssistant()
STATIC_DIR = ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------
class AccountCreate(BaseModel):
    id: str
    username: str
    token_dir: Optional[str] = None


class AccountTokenUpdate(BaseModel):
    token: Optional[str] = None
    username: Optional[str] = None


class JobCreate(BaseModel):
    project_name: str
    name: str
    account_id: str
    job_type: str = "benchmark"
    engines: List[str] = ["paddleocr"]
    install_files: List[str] = ["requirements-kaggle-paddleocr.txt"]
    machine_shape: str = "NvidiaTeslaT4"
    finetune_model: Optional[str] = "paddleocr"
    epochs: Optional[int] = 10
    limit: Optional[int] = 20


class RunJobsRequest(BaseModel):
    project_name: str
    job_names: Optional[List[str]] = None


class ChatRequest(BaseModel):
    message: str
    project_name: Optional[str] = None


class GitCommitRequest(BaseModel):
    message: str


# ---------------------------------------------------------
# Accounts API
# ---------------------------------------------------------
@app.get("/api/accounts")
def get_accounts(refresh: bool = False):
    accounts_cfg = load_accounts()
    accounts_list = accounts_cfg.get("accounts", [])
    if not accounts_list:
        return {"accounts": []}

    def _fetch_status(acc):
        status = get_account_gpu_status(acc, force_refresh=refresh)
        return {
            "id": acc["id"],
            "username": acc.get("username", ""),
            "token_dir": acc.get("token_dir", f"data/tokens/{acc['id']}"),
            "token_valid": status.get("token_valid", False),
            "active_sessions": status.get("active_sessions", 0),
            "max_sessions": status.get("max_sessions", 2),
            "gpu_hours_used": status.get("gpu_hours_used", 0.0),
            "gpu_hours_total": status.get("gpu_hours_total", 30.0),
            "gpu_hours_remaining": status.get("gpu_hours_remaining", 30.0),
            "running_kernels": status.get("running_kernels", []),
            "message": status.get("message", ""),
        }

    max_workers = min(len(accounts_list), 8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_fetch_status, accounts_list))

    return {"accounts": results}


@app.post("/api/accounts")
def create_account(data: AccountCreate):
    ok = add_account(data.id.strip(), data.username.strip(), data.token_dir)
    if not ok:
        raise HTTPException(status_code=400, detail="Account with this ID already exists.")
    clear_api_cache(data.id.strip())
    assistant.refresh()
    return {"success": True, "message": f"Account {data.id} created successfully."}


@app.delete("/api/accounts/{account_id}")
def remove_account(account_id: str):
    delete_account(account_id)
    clear_api_cache(account_id)
    assistant.refresh()
    return {"success": True, "message": f"Account {account_id} removed."}


@app.post("/api/accounts/{account_id}/validate")
def validate_acc(account_id: str, refresh: bool = True):
    accounts = accounts_by_id()
    account = accounts.get(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found.")
    res = validate_account(account, force_refresh=refresh)
    return res


@app.post("/api/accounts/{account_id}/token")
def update_token(account_id: str, data: AccountTokenUpdate):
    accounts = accounts_by_id()
    account = accounts.get(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found.")
    token_dir = resolve_tool_path(account["token_dir"])
    username = data.username or account.get("username", "user")
    if data.token:
        write_kaggle_json(token_dir, username, data.token)
    clear_api_cache(account_id)
    return {"success": True, "message": "API Token updated successfully."}


@app.post("/api/accounts/{account_id}/upload_kaggle_json")
async def upload_kaggle_json_file(account_id: str, file: UploadFile = File(...)):
    accounts = accounts_by_id()
    account = accounts.get(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found.")
    token_dir = resolve_tool_path(account["token_dir"])
    token_dir.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    (token_dir / "kaggle.json").write_bytes(content)
    clear_api_cache(account_id)
    return {"success": True, "message": "kaggle.json saved successfully."}


# ---------------------------------------------------------
# Projects & Jobs API (Strategy 2: Async Concurrent Execution)
# ---------------------------------------------------------
@app.get("/api/projects")
def get_all_projects():
    return load_projects()


@app.get("/api/projects/{project_name}/jobs_status")
async def get_jobs_status(project_name: str, force_refresh: bool = False):
    projects_cfg = load_projects()
    accounts = accounts_by_id()
    project = next((p for p in projects_cfg.get("projects", []) if p["name"] == project_name), None)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    jobs = project.get("jobs", [])

    async def _fetch_job_status(job: Dict[str, Any]) -> Dict[str, Any]:
        acc = accounts.get(job["account_id"])
        k_id = kernel_id(acc, job) if acc else "N/A"
        st_text = "UNKNOWN"
        raw_output = ""
        if acc:
            res = await asyncio.to_thread(status_job, acc, job, force_refresh)
            raw_output = (getattr(res, "stdout", "") + getattr(res, "stderr", "")).strip()
            raw_upper = raw_output.upper()
            if "COMPLETE" in raw_upper:
                st_text = "COMPLETE"
            elif "RUNNING" in raw_upper:
                st_text = "RUNNING"
            elif "QUEUED" in raw_upper:
                st_text = "QUEUED"
            elif "ERROR" in raw_upper or "FAILED" in raw_upper:
                st_text = "ERROR"
            else:
                st_text = "IDLE"

        return {
            "name": job["name"],
            "account_id": job["account_id"],
            "username": acc.get("username", "N/A") if acc else "N/A",
            "kernel_id": k_id,
            "job_type": job.get("job_type", "benchmark"),
            "engines": job.get("engines", []),
            "install_files": job.get("install_files", []),
            "machine_shape": job.get("machine_shape", "NvidiaTeslaT4"),
            "status": st_text,
            "raw_output": raw_output,
        }

    if jobs:
        jobs_data = await asyncio.gather(*[_fetch_job_status(job) for job in jobs])
    else:
        jobs_data = []

    return {"project": project_name, "jobs": list(jobs_data)}


@app.post("/api/jobs/add")
def add_new_job(data: JobCreate):
    job_dict = {
        "name": data.name.strip(),
        "account_id": data.account_id.strip(),
        "job_type": data.job_type,
        "machine_shape": data.machine_shape,
        "engines": data.engines,
        "install_files": data.install_files,
    }
    if data.job_type == "finetune":
        job_dict["finetune_model"] = data.finetune_model or "paddleocr"
        job_dict["epochs"] = data.epochs or 10

    ok = add_job_to_project(data.project_name, job_dict)
    if not ok:
        raise HTTPException(status_code=400, detail="Thread already exists or project not found.")
    return {"success": True, "message": f"Thread {data.name} created."}


@app.delete("/api/jobs/{project_name}/{job_name}")
def remove_job(project_name: str, job_name: str):
    delete_job_from_project(project_name, job_name)
    return {"success": True, "message": f"Thread {job_name} deleted."}


@app.post("/api/jobs/run")
async def trigger_run_jobs(data: RunJobsRequest, background_tasks: BackgroundTasks):
    projects_cfg = load_projects()
    accounts = accounts_by_id()
    project = next((p for p in projects_cfg.get("projects", []) if p["name"] == data.project_name), None)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    prepared = prepare_project_jobs(project, accounts)
    if data.job_names:
        prepared = [it for it in prepared if it["job"]["name"] in data.job_names]

    if not prepared:
        raise HTTPException(status_code=400, detail="No valid threads to execute.")

    push_tasks = [(push_job, (it["account"], it["job_dir"])) for it in prepared]
    results = await asyncio.to_thread(run_parallel_tasks, push_tasks, len(prepared) or 1)

    dispatched = []
    for it, res in zip(prepared, results):
        dispatched.append({
            "job": it["job"]["name"],
            "kernel_id": it["kernel_id"],
            "stdout": getattr(res, "stdout", str(res)),
            "stderr": getattr(res, "stderr", ""),
        })

    return {"success": True, "count": len(dispatched), "jobs": dispatched}


@app.post("/api/jobs/stop")
async def stop_single_job(project_name: str = Form(...), job_name: str = Form(...)):
    projects_cfg = load_projects()
    accounts = accounts_by_id()
    project = next((p for p in projects_cfg.get("projects", []) if p["name"] == project_name), None)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    job = next((j for j in project.get("jobs", []) if j["name"] == job_name), None)
    if not job:
        raise HTTPException(status_code=404, detail="Thread not found.")

    acc = accounts.get(job["account_id"])
    if not acc:
        raise HTTPException(status_code=400, detail="Account not found for this thread.")

    res = await asyncio.to_thread(stop_job, acc, job)
    msg = (getattr(res, "stdout", "") + getattr(res, "stderr", "")).strip() or "Stop signal dispatched."
    return {"success": True, "message": msg}


@app.post("/api/jobs/download")
async def download_artifacts(data: RunJobsRequest):
    projects_cfg = load_projects()
    accounts = accounts_by_id()
    project = next((p for p in projects_cfg.get("projects", []) if p["name"] == data.project_name), None)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    jobs = project.get("jobs", [])
    if data.job_names:
        jobs = [j for j in jobs if j["name"] in data.job_names]

    async def _download_one(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        acc = accounts.get(job["account_id"])
        if acc:
            out_dir, res = await asyncio.to_thread(download_job_output, project["name"], acc, job)
            return {"job": job["name"], "dir": str(out_dir), "code": getattr(res, "returncode", 0)}
        return None

    downloaded = await asyncio.gather(*[_download_one(job) for job in jobs])
    downloaded = [d for d in downloaded if d is not None]

    return {"success": True, "downloaded": downloaded}


# ---------------------------------------------------------
# Analytics & Reports API
# ---------------------------------------------------------
@app.get("/api/analytics/{project_name}")
async def get_analytics(project_name: str):
    results_dir = OUTPUTS_DIR / project_name
    fallback_dir = ROOT.parent / "kaggle_remote_jobs" / "outputs"
    target_dir = results_dir
    is_main_project = project_name in {"pdf_ocr_benchmark", "default"}

    combined_raw = await asyncio.to_thread(combine_summaries, target_dir, deduplicate_engine=False)
    combined = await asyncio.to_thread(combine_summaries, target_dir, deduplicate_engine=True)
    if combined.empty and is_main_project and fallback_dir.exists():
        combined_raw = await asyncio.to_thread(combine_summaries, fallback_dir, deduplicate_engine=False)
        combined = await asyncio.to_thread(combine_summaries, fallback_dir, deduplicate_engine=True)

    finetune_df = await asyncio.to_thread(collect_finetune_status, target_dir)
    if finetune_df.empty and is_main_project and fallback_dir.exists():
        finetune_df = await asyncio.to_thread(collect_finetune_status, fallback_dir)

    comparison_data = []
    if not combined_raw.empty:
        comparison_data = build_comparison_table(combined_raw)

    summary_data = json.loads(combined.to_json(orient="records")) if not combined.empty else []
    finetune_data = json.loads(finetune_df.to_json(orient="records")) if not finetune_df.empty else []

    has_data = len(summary_data) > 0 or len(finetune_data) > 0
    return {
        "summary": summary_data,
        "finetune": finetune_data,
        "comparison": comparison_data,
        "has_data": has_data,
    }


# ---------------------------------------------------------
# Logs API (Strategy 4: Fast Tail Reader)
# ---------------------------------------------------------
@app.get("/api/logs/{project_name}")
async def get_logs(project_name: str, file_path: Optional[str] = None, max_lines: int = 150):
    project_output_dir = OUTPUTS_DIR / project_name
    if not project_output_dir.exists():
        return {"files": [], "content": "No log files available."}

    files_dict = await asyncio.to_thread(find_job_files, project_output_dir)
    log_files = files_dict.get("logs", [])

    files_info = [{"path": str(p), "name": p.name, "job": p.parent.name} for p in log_files]

    content = ""
    target_file = None
    if file_path:
        target_file = Path(file_path)
    elif log_files:
        target_file = log_files[0]

    if target_file and target_file.exists():
        content = await asyncio.to_thread(read_text_tail, target_file, max_lines=max_lines)

    return {"files": files_info, "selected_file": str(target_file) if target_file else "", "content": content}


# ---------------------------------------------------------
# AI Assistant & Chat API
# ---------------------------------------------------------
@app.post("/api/chat")
async def handle_chat(data: ChatRequest):
    res = await asyncio.to_thread(assistant.process_message, data.message, project_name=data.project_name)
    return res


# ---------------------------------------------------------
# Git Control API
# ---------------------------------------------------------
@app.get("/api/git/status")
async def git_status():
    def _run_git():
        res = subprocess.run(["git", "status", "--short"], cwd=str(ROOT.parent), capture_output=True, text=True)
        branch_res = subprocess.run(["git", "branch", "--show-current"], cwd=str(ROOT.parent), capture_output=True, text=True)
        return {"branch": branch_res.stdout.strip(), "status": res.stdout.strip()}

    return await asyncio.to_thread(_run_git)


@app.post("/api/git/commit")
async def git_commit(data: GitCommitRequest):
    def _run_commit():
        subprocess.run(["git", "add", "."], cwd=str(ROOT.parent), check=False)
        res = subprocess.run(["git", "commit", "-m", data.message], cwd=str(ROOT.parent), capture_output=True, text=True)
        return {"output": res.stdout + res.stderr}

    return await asyncio.to_thread(_run_commit)


@app.post("/api/git/push")
async def git_push():
    def _run_push():
        res = subprocess.run(["git", "push"], cwd=str(ROOT.parent), capture_output=True, text=True)
        return {"output": res.stdout + res.stderr}

    return await asyncio.to_thread(_run_push)


# ---------------------------------------------------------
# Static Frontend Serving
# ---------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Kaggle Multi-Account Hub</h1><p>Building frontend...</p>")


def main():
    parser = argparse.ArgumentParser(description="Kaggle Multi-Account Control Hub Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--open-browser", action="store_true", default=True)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print(f"\n=======================================================")
    print(f"Kaggle Multi-Account Hub server running at: {url}")
    print(f"=======================================================\n")

    if args.open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
