import concurrent.futures
import json
import shlex
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader

from .kaggle_api import (
    cancel_kernel,
    download_kernel_output,
    push_kernel,
    query_kernel_status,
    run_kaggle,
)
from .project_config import ROOT, resolve_tool_path

WORK_DIR = ROOT / "work"
OUTPUTS_DIR = ROOT / "outputs"
TEMPLATES_DIR = ROOT / "templates"


def job_slug(name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "-" for ch in name.lower()).strip("-")
    if not slug.startswith("uwir-"):
        slug = f"uwir-{slug}"
    return slug[:50]


def kernel_id(account: Dict[str, Any], job: Dict[str, Any]) -> str:
    return f"{account['username']}/{job_slug(job['name'])}"


def prepare_project_jobs(project: Dict[str, Any], accounts: Dict[str, Any]) -> List[Dict[str, Any]]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    prepared = []
    for job in project.get("jobs", []):
        account = accounts.get(job["account_id"])
        if not account:
            continue
        slug = job_slug(job["name"])
        job_dir = WORK_DIR / project["name"] / slug
        if job_dir.exists():
            shutil.rmtree(job_dir)
        job_dir.mkdir(parents=True)
        code_file = f"{slug}.py"
        context = build_context(project, job, account, slug, code_file)
        env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=False)
        metadata = env.get_template("kernel-metadata.json.j2").render(**context)
        script = env.get_template("kernel_script.py.j2").render(**context)
        (job_dir / "kernel-metadata.json").write_text(metadata, encoding="utf-8")
        (job_dir / code_file).write_text(script, encoding="utf-8")
        prepared.append({
            "job": job,
            "account": account,
            "job_dir": job_dir,
            "kernel_id": kernel_id(account, job),
            "slug": slug,
        })
    return prepared


def build_context(
    project: Dict[str, Any],
    job: Dict[str, Any],
    account: Dict[str, Any],
    slug: str,
    code_file: str,
) -> Dict[str, Any]:
    branch = job.get("branch", project.get("branch", "feat/pcf-mbconv-fusion"))
    repo_url = job.get("repo_url", project.get("repo_url", "https://github.com/heniath/underwater-image-enhancement.git"))
    args_file = job.get("args_file", "configs/kaggle_pcf_mbconv_5ch_euvp.args")
    dataset_sources = job.get(
        "dataset_sources",
        project.get("dataset_sources", ["pamuduranasinghe/euvp-dataset"])
    )

    setup_commands = job.get("setup_commands") or project.get("commands", {}).get("setup") or [
        "python -m pip install -q -e .",
        "python -m pip install -q -r requirements.txt",
    ]

    raw_run_commands = job.get("run_commands") or project.get("commands", {}).get("run") or [
        f"python train.py @{args_file} --data_train_euvp $EUVP_ROOT"
    ]

    variables = {
        "slug": slug,
        "branch": branch,
        "args_file": args_file,
        "result_dir": f"/kaggle/working/results_{slug}",
    }
    rendered_run_commands = [render_command(command, variables) for command in raw_run_commands]
    install_commands = [f"python -m pip install -q -r {path}" for path in job.get("install_files", [])]

    local_code_dir = job.get("local_code_dir") or project.get("local_code_dir")
    code_bundle_b64 = ""
    if local_code_dir:
        dir_p = resolve_tool_path(local_code_dir)
        src_p = dir_p / "src"
        if src_p.exists():
            import base64
            import io
            import tarfile
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                tar.add(str(src_p), arcname="src")
            code_bundle_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "username": account["username"],
        "slug": slug,
        "code_file": code_file,
        "branch": branch,
        "repo_url": repo_url,
        "machine_shape": job.get("machine_shape", project.get("machine_shape", "NvidiaTeslaT4")),
        "dataset_sources": dataset_sources,
        "code_bundle_b64": code_bundle_b64,
        "setup_commands": [command_to_list(cmd) for cmd in setup_commands],
        "install_commands": [command_to_list(cmd) for cmd in install_commands],
        "run_commands": [command_to_list(cmd) for cmd in rendered_run_commands],
    }


def render_command(command: str, variables: Dict[str, Any]) -> str:
    for key, value in variables.items():
        command = command.replace("{{ " + key + " }}", str(value)).replace("{{" + key + "}}", str(value))
    return command


def command_to_list(command: str) -> str:
    return json.dumps(shlex.split(command))


def push_job(account: Dict[str, Any], job_dir: Path) -> Any:
    return push_kernel(account, job_dir)


def status_job(account: Dict[str, Any], job: Dict[str, Any], force_refresh: bool = False) -> Any:
    return query_kernel_status(account, kernel_id(account, job), force_refresh=force_refresh)


def stop_job(account: Dict[str, Any], job: Dict[str, Any]) -> Any:
    k_id = kernel_id(account, job)
    return cancel_kernel(account, k_id)


def download_job_output(project_name: str, account: Dict[str, Any], job: Dict[str, Any]) -> Tuple[Path, Any]:
    out_dir = OUTPUTS_DIR / project_name / job["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    res = download_kernel_output(account, kernel_id(account, job), out_dir)
    return out_dir, res


def run_parallel_tasks(tasks: List[Tuple[Callable, Tuple]], max_workers: int = 4) -> List[Any]:
    """Executes multiple Kaggle operations concurrently across threads."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(fn, *args): (fn, args) for fn, args in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(exc)
    return results
