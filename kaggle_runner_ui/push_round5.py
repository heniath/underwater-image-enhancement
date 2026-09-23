#!/usr/bin/env python3
import sys
import time
from pathlib import Path

# Add kaggle_runner_ui to sys.path
runner_dir = Path(__file__).resolve().parent
if str(runner_dir) not in sys.path:
    sys.path.insert(0, str(runner_dir))

from runner.project_config import accounts_by_id, get_project
from runner.job_builder import prepare_project_jobs, push_job, status_job

TARGET_JOBS = [
    "sgmanet-combo-lvw2-uiqm02-uieb800",
    "sgmanet-loss-lvw-w5-uieb800",
    "sgmanet-loss-lvw-w10-uieb800",
    "sgmanet-loss-lvw-w30-uieb800",
    "sgmanet-loss-lvw-w50-uieb800",
    "sgmanet-combo-lvw2-uiqm01-uieb800",
]

def main():
    print("Preparing project jobs for sgmanet_loss_ablations...")
    accounts = accounts_by_id()
    project = get_project("sgmanet_loss_ablations")
    if not project:
        print("Project sgmanet_loss_ablations not found!")
        sys.exit(1)

    all_prepared = prepare_project_jobs(project, accounts)
    print(f"Total prepared jobs: {len(all_prepared)}")

    jobs_to_push = [j for j in all_prepared if j["job"]["name"] in TARGET_JOBS]
    print(f"Filtered {len(jobs_to_push)} jobs for Round 5:")
    for j in jobs_to_push:
        print(f" - {j['job']['name']} -> Account: {j['account']['username']} ({j['account']['id']}) | Dir: {j['job_dir']}")

    print("\nPushing Round 5 kernels to Kaggle GPU...")
    push_results = {}
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
        push_results[name] = res.returncode == 0
        time.sleep(2)  # Short pause between pushes

    print("\nWaiting 10 seconds before verifying initial status...")
    time.sleep(10)

    print("\n--- Current Status of Round 5 Jobs ---")
    for j in jobs_to_push:
        name = j["job"]["name"]
        acc = j["account"]
        st = status_job(acc, j["job"], force_refresh=True)
        print(f"{name} ({acc['username']}): {st.stdout.strip() or st.stderr.strip()}")

if __name__ == "__main__":
    main()
