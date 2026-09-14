import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .job_builder import OUTPUTS_DIR, download_job_output, kernel_id, prepare_project_jobs, push_job, status_job
from .kaggle_api import get_account_gpu_status, validate_account
from .log_reader import combine_summaries, find_job_files, read_csv, read_text_tail
from .project_config import ROOT, accounts_by_id, load_accounts, load_projects


class KaggleAIAssistant:
    """
    Antigravity AI Assistant Companion for Kaggle Multi-Account & Multi-Job Management.
    Processes natural language commands, automates dispatching, monitors GPU usage,
    and analyzes OCR benchmark logs.
    """

    def __init__(self):
        self.accounts_cfg = load_accounts()
        self.projects_cfg = load_projects()
        self.accounts = accounts_by_id()

    def refresh(self):
        self.accounts_cfg = load_accounts()
        self.projects_cfg = load_projects()
        self.accounts = accounts_by_id()

    def process_message(self, user_prompt: str, project_name: Optional[str] = None) -> Dict[str, Any]:
        self.refresh()
        prompt_lower = user_prompt.lower().strip()
        project = None
        projects = self.projects_cfg.get("projects", [])
        if project_name:
            project = next((p for p in projects if p["name"] == project_name), None)
        if not project and projects:
            project = projects[0]

        # 1. Quota / GPU check
        if any(k in prompt_lower for k in ["gpu", "quota", "usage", "slot", "account", "capacity", "hours"]):
            return self._handle_quota_query()

        # 2. Status check
        if any(k in prompt_lower for k in ["status", "progress", "running", "jobs", "threads", "check"]):
            return self._handle_status_query(project)

        # 3. Download results / Report
        if any(k in prompt_lower for k in ["download", "results", "report", "summary", "fetch"]):
            return self._handle_download_query(project)

        # 4. Error / Log Analysis
        if any(k in prompt_lower for k in ["error", "fail", "debug", "why", "log", "traceback", "diagnose"]):
            return self._handle_error_diagnosis(project)

        # 5. Run commands
        if any(k in prompt_lower for k in ["run", "start", "train", "finetune", "execute", "launch"]):
            return self._handle_run_query(prompt_lower, project)

        # 6. Default response with actions
        return {
            "response": (
                "**Antigravity Assistant Ready.**\n\n"
                "Available automated actions:\n"
                "- **GPU Quota & Active Slots:** 'Check GPU quota for all accounts'\n"
                "- **Multi-Thread Control:** 'Run all threads' or 'Run paddleocr on account2'\n"
                "- **Progress Monitoring:** 'Check current thread status'\n"
                "- **Results & Reports:** 'Download results and generate summary'\n"
                "- **Log Diagnostics:** 'Analyze recent error logs'\n\n"
                "What would you like to execute?"
            ),
            "actions": [
                {"label": "Check GPU Quota", "command": "Check GPU quota for all accounts"},
                {"label": "Check Thread Status", "command": "Check current thread status"},
                {"label": "Run All Threads", "command": "Run all threads"},
                {"label": "Download Results", "command": "Download results and generate summary"},
            ],
        }

    def _handle_quota_query(self) -> Dict[str, Any]:
        accounts_list = self.accounts_cfg.get("accounts", [])
        if not accounts_list:
            return {"response": "No Kaggle accounts configured in the system."}

        lines = ["### Kaggle Accounts & GPU Quota Report\n"]
        lines.append("| Account | Username | Active GPU Slots | Weekly GPU Quota | Token Status |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")

        total_active = 0
        for acc in accounts_list:
            status = get_account_gpu_status(acc)
            active_slots = f"{status.get('active_sessions', 0)} / {status.get('max_sessions', 2)} slots"
            total_active += status.get("active_sessions", 0)
            hours_str = f"{status.get('gpu_hours_used', 0.0):.1f}h / {status.get('gpu_hours_total', 30.0):.0f}h"
            token_valid = "Valid" if status.get("token_valid") else "Invalid / Missing"
            lines.append(f"| **{acc['id']}** | `{acc.get('username', 'N/A')}` | `{active_slots}` | `{hours_str}` | {token_valid} |")

        lines.append(f"\n> **Summary:** Currently **{total_active} active GPU session(s)** across {len(accounts_list) * 2} total slots.")
        lines.append("> *Note: Kaggle limits each account to 30 GPU hours/week and a maximum of 2 concurrent batch GPU sessions.*")

        return {
            "response": "\n".join(lines),
            "actions": [
                {"label": "View Active Threads", "command": "Check current thread status"},
                {"label": "Manage Accounts", "command": "Open Settings & Git tab"},
            ],
        }

    def _handle_status_query(self, project: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not project:
            return {"response": "No active project found."}

        jobs = project.get("jobs", [])
        if not jobs:
            return {"response": f"Project '{project['name']}' has no configured jobs."}

        lines = [f"### Execution Status for Project `{project['name']}`\n"]
        lines.append("| Job Name | Account | Kernel ID | Status | Engines |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")

        for job in jobs:
            account = self.accounts.get(job["account_id"])
            if not account:
                lines.append(f"| **{job['name']}** | `{job['account_id']}` | N/A | Missing Account | {', '.join(job.get('engines', []))} |")
                continue

            k_id = kernel_id(account, job)
            res = status_job(account, job)
            status_text = (res.stdout + res.stderr).strip()
            badge = "IDLE"
            if "COMPLETE" in status_text.upper():
                badge = "**COMPLETE**"
            elif "RUNNING" in status_text.upper():
                badge = "**RUNNING**"
            elif "QUEUED" in status_text.upper():
                badge = "**QUEUED**"
            elif "ERROR" in status_text.upper() or "FAILED" in status_text.upper():
                badge = "**ERROR**"

            lines.append(f"| **{job['name']}** | `{account['username']}` | `{k_id}` | {badge} | {', '.join(job.get('engines', []))} |")

        return {
            "response": "\n".join(lines),
            "actions": [
                {"label": "Download Completed Results", "command": "Download results and generate summary"},
                {"label": "Analyze Error Logs", "command": "Analyze recent error logs"},
            ],
        }

    def _handle_download_query(self, project: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not project:
            return {"response": "No active project found."}

        results_dir = OUTPUTS_DIR / project["name"]
        combined = combine_summaries(results_dir) if results_dir.exists() else None

        if combined is None or combined.empty:
            return {
                "response": (
                    f"No aggregated `summary.csv` found locally for project `{project['name']}` yet. "
                    "Use the **Download All** action to fetch all artifacts from Kaggle."
                ),
                "actions": [{"label": "Download Now", "command": "Download results and generate summary"}],
            }

        lines = [f"### Latest Benchmark Summary (`{project['name']}`)\n"]
        lines.append(combined.to_markdown(index=False))
        return {
            "response": "\n".join(lines),
            "actions": [
                {"label": "View Analytics Charts", "command": "Open Benchmark Analytics tab"},
                {"label": "View Execution Logs", "command": "Open Execution Logs tab"},
            ],
        }

    def _handle_error_diagnosis(self, project: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not project:
            return {"response": "No active project found."}

        output_dir = OUTPUTS_DIR / project["name"]
        if not output_dir.exists():
            return {"response": "No local log files found. Please download artifacts from Kaggle first."}

        files = find_job_files(output_dir)
        log_files = files.get("logs", [])
        if not log_files:
            return {"response": f"No `.log` files found in output directory for project '{project['name']}'."}

        diagnoses = []
        for log_path in log_files:
            content = read_text_tail(log_path, max_lines=80)
            if "Traceback" in content or "Error" in content or "FAILED" in content:
                job_name = log_path.parent.name if log_path.parent.name != project["name"] else log_path.stem
                error_match = re.findall(r"(?:[A-Za-z]+Error:.*|Traceback.*)", content, re.MULTILINE)
                summary_err = "\n".join(error_match[-3:]) if error_match else "Runtime exception occurred."
                diagnoses.append(f"#### Thread `{job_name}` (`{log_path.name}`)\n```text\n{summary_err}\n```")

        if not diagnoses:
            return {
                "response": "No critical errors found in recent log files. All threads completed successfully or are executing normally."
            }

        response_text = (
            "### Log Analysis & Diagnostic Results:\n\n"
            + "\n\n".join(diagnoses)
            + "\n\n**Recommended Actions:**\n"
            + "1. Verify and update the corresponding `requirements-kaggle-*.txt` dependencies.\n"
            + "2. Re-trigger the thread from the Thread Jobs tab after pushing updates."
        )

        return {
            "response": response_text,
            "actions": [
                {"label": "View Full Log", "command": "Open Execution Logs tab"},
                {"label": "Retry Failed Threads", "command": "Run all threads"},
            ],
        }

    def _handle_run_query(self, prompt_lower: str, project: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not project:
            return {"response": "No active project found."}

        jobs = project.get("jobs", [])
        target_jobs = []
        for job in jobs:
            if job["name"].lower() in prompt_lower:
                target_jobs.append(job)

        if not target_jobs and ("all" in prompt_lower or "every" in prompt_lower):
            target_jobs = jobs

        if not target_jobs:
            job_names = ", ".join([f"`{j['name']}`" for j in jobs])
            return {
                "response": f"Which thread would you like to run? Available: {job_names}. Or specify 'Run all threads'.",
                "actions": [{"label": "Run All Threads", "command": "Run all threads"}],
            }

        job_list_str = ", ".join([f"**{j['name']}** (Account: `{j['account_id']}`)" for j in target_jobs])
        return {
            "response": (
                f"Ready to dispatch {len(target_jobs)} thread(s):\n"
                f"{job_list_str}\n\n"
                "Click the action below or use the Thread Jobs tab to initiate parallel execution."
            ),
            "action_trigger": "run_jobs",
            "target_job_names": [j["name"] for j in target_jobs],
            "actions": [
                {"label": f"Dispatch {len(target_jobs)} Thread(s) Now", "command": f"Execute {len(target_jobs)} thread(s)"},
                {"label": "Check Quotas First", "command": "Check GPU quota for all accounts"},
            ],
        }
