from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ACCOUNTS_PATH = DATA_DIR / "accounts.yaml"
PROJECTS_PATH = DATA_DIR / "projects.yaml"
ACCOUNTS_EXAMPLE_PATH = DATA_DIR / "accounts.example.yaml"
PROJECTS_EXAMPLE_PATH = DATA_DIR / "projects.example.yaml"


def ensure_data_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ACCOUNTS_PATH.exists():
        if ACCOUNTS_EXAMPLE_PATH.exists():
            ACCOUNTS_PATH.write_text(ACCOUNTS_EXAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            ACCOUNTS_PATH.write_text("accounts: []\n", encoding="utf-8")
    if not PROJECTS_PATH.exists():
        if PROJECTS_EXAMPLE_PATH.exists():
            PROJECTS_PATH.write_text(PROJECTS_EXAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            PROJECTS_PATH.write_text("projects: []\n", encoding="utf-8")


def load_accounts() -> Dict[str, Any]:
    ensure_data_files()
    return yaml.safe_load(ACCOUNTS_PATH.read_text(encoding="utf-8")) or {"accounts": []}


def save_accounts(config: Dict[str, Any]):
    ensure_data_files()
    ACCOUNTS_PATH.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def load_projects() -> Dict[str, Any]:
    ensure_data_files()
    return yaml.safe_load(PROJECTS_PATH.read_text(encoding="utf-8")) or {"projects": []}


def save_projects(config: Dict[str, Any]):
    ensure_data_files()
    PROJECTS_PATH.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def accounts_by_id() -> Dict[str, Any]:
    return {account["id"]: account for account in load_accounts().get("accounts", [])}


def get_project(name: str) -> Optional[Dict[str, Any]]:
    for project in load_projects().get("projects", []):
        if project.get("name") == name:
            return project
    return None


def add_account(account_id: str, username: str, token_dir: Optional[str] = None) -> bool:
    cfg = load_accounts()
    accounts = cfg.get("accounts", [])
    if any(a["id"] == account_id for a in accounts):
        return False
    if not token_dir:
        token_dir = f"data/tokens/{account_id}"
    accounts.append({"id": account_id, "username": username, "token_dir": token_dir})
    save_accounts({"accounts": accounts})
    return True


def delete_account(account_id: str) -> bool:
    cfg = load_accounts()
    accounts = [a for a in cfg.get("accounts", []) if a["id"] != account_id]
    save_accounts({"accounts": accounts})
    return True


def add_job_to_project(project_name: str, job_dict: Dict[str, Any]) -> bool:
    cfg = load_projects()
    projects = cfg.get("projects", [])
    project = next((p for p in projects if p["name"] == project_name), None)
    if not project:
        return False
    jobs = project.get("jobs", [])
    if any(j["name"] == job_dict["name"] for j in jobs):
        return False
    jobs.append(job_dict)
    project["jobs"] = jobs
    save_projects({"projects": projects})
    return True


def delete_job_from_project(project_name: str, job_name: str) -> bool:
    cfg = load_projects()
    projects = cfg.get("projects", [])
    project = next((p for p in projects if p["name"] == project_name), None)
    if not project:
        return False
    project["jobs"] = [j for j in project.get("jobs", []) if j["name"] != job_name]
    save_projects({"projects": projects})
    return True


def token_file_paths(token_dir: Any):
    token_dir = resolve_tool_path(token_dir)
    return token_dir / "kaggle.json", token_dir / "access_token"


def resolve_tool_path(path: Any) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
