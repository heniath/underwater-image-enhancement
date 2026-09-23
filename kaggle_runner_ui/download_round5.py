#!/usr/bin/env python3
import sys
import time
from pathlib import Path

runner_dir = Path(__file__).resolve().parent
if str(runner_dir) not in sys.path:
    sys.path.insert(0, str(runner_dir))

from runner.project_config import accounts_by_id, get_project
from runner.job_builder import download_job_output, kernel_id
from runner.kaggle_api import run_kaggle

ROUND5_JOBS = [
    "sgmanet-combo-lvw2-uiqm02-uieb800",
    "sgmanet-loss-lvw-w5-uieb800",
    "sgmanet-loss-lvw-w10-uieb800",
    "sgmanet-loss-lvw-w30-uieb800",
    "sgmanet-loss-lvw-w50-uieb800",
    "sgmanet-combo-lvw2-uiqm01-uieb800",
]

def main():
    print("Starting download for Round 5 jobs...")
    accounts = accounts_by_id()
    project = get_project("sgmanet_loss_ablations")
    if not project:
        print("Project not found!")
        sys.exit(1)

    jobs_by_name = {j["name"]: j for j in project["jobs"]}

    for name in ROUND5_JOBS:
        job = jobs_by_name.get(name)
        if not job:
            print(f"Job {name} not found in project config!")
            continue

        acc = accounts.get(job["account_id"])
        k_id = kernel_id(acc, job)
        print(f"\n=======================================================")
        print(f"Downloading outputs for {name} ({k_id})...")
        out_dir, res = download_job_output("sgmanet_loss_ablations", acc, job)
        print(f"Download result code: {res.returncode}")
        if res.stdout:
            print(f"stdout:\n{res.stdout.strip()[:1000]}")
        if res.stderr:
            print(f"stderr:\n{res.stderr.strip()[:1000]}")

        # Check what files were downloaded
        downloaded_files = list(out_dir.glob("**/*"))
        print(f"Total items in {out_dir}: {len(downloaded_files)}")
        for f in downloaded_files:
            if f.is_file():
                print(f" - {f.relative_to(out_dir)} ({f.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
