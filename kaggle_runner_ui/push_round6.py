#!/usr/bin/env python3
import sys
import time
from pathlib import Path

runner_dir = Path(__file__).resolve().parent
if str(runner_dir) not in sys.path:
    sys.path.insert(0, str(runner_dir))

from runner.project_config import accounts_by_id, get_project
from runner.job_builder import prepare_project_jobs, push_job, status_job

TARGET_JOBS = [
    "sgmanet-combo-lvw5-uiqm02-uieb800",
    "sgmanet-combo-lvw5-uiqm1-uieb800",
    "sgmanet-combo-lvw5-uiqm01-uieb800",
]

def push_selected(selected_names=None):
    if selected_names is None:
        selected_names = TARGET_JOBS
    accounts = accounts_by_id()
    project = get_project("sgmanet_loss_ablations")
    if not project:
        print("Project sgmanet_loss_ablations not found!")
        sys.exit(1)

    print("Preparing project jobs...")
    all_prepared = prepare_project_jobs(project, accounts)
    jobs_to_push = [j for j in all_prepared if j["job"]["name"] in selected_names]
    print(f"Selected {len(jobs_to_push)} jobs to push:")
    for j in jobs_to_push:
        print(f" - {j['job']['name']} -> Account: {j['account']['username']} ({j['account']['id']})")

    for j in jobs_to_push:
        name = j["job"]["name"]
        acc = j["account"]
        job_dir = j["job_dir"]
        print(f"\n--> Pushing {name} to {acc['username']}...")
        res = push_job(acc, job_dir)
        print(f"Result for {name}: code={res.returncode}")
        if res.stdout:
            print(f"  stdout: {res.stdout.strip()}")
        if res.stderr:
            print(f"  stderr: {res.stderr.strip()}")
        time.sleep(2)

    print("\nWaiting 10 seconds before verifying initial status...")
    time.sleep(10)
    for j in jobs_to_push:
        name = j["job"]["name"]
        acc = j["account"]
        st = status_job(acc, j["job"], force_refresh=True)
        print(f"{name} ({acc['username']}): {st.stdout.strip() or st.stderr.strip()}")

if __name__ == "__main__":
    selected = sys.argv[1:] if len(sys.argv) > 1 else TARGET_JOBS
    push_selected(selected)
