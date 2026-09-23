#!/usr/bin/env python3
"""
auto_dispatcher.py
------------------
Autonomous queue manager and output downloader for Kaggle loss ablation jobs.
Monitors running jobs and auto-dispatches queued experiments (SSIM on TThanh13,
HVI on RaH1111) as soon as GPU slots become available.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import requests

SERVER_URL = "http://localhost:8080"
PROJECT_NAME = "sgmanet_loss_ablations"
POLL_INTERVAL = 30  # seconds

# Desired queue mappings
PENDING_DISPATCH = {
    "sgmanet-loss-ssim-uieb800": {"account_id": "account3", "user": "RaH1111", "dispatched": True},
    "sgmanet-loss-hvi-uieb800": {"account_id": "account3", "user": "RaH1111", "dispatched": True},
    "sgmanet-loss-lappyr-uieb800": {"account_id": "account3", "user": "RaH1111", "dispatched": True},
}

DOWNLOADED_JOBS = {
    "sgmanet-loss-lvw-uieb800",
    "sgmanet-loss-edge-uieb800",
    "sgmanet-loss-uiqm-uieb800",
    "sgmanet-loss-base-uieb800",
    "sgmanet-loss-ssim-uieb800",
    "sgmanet-loss-hvi-uieb800",
    "sgmanet-loss-lappyr-uieb800",
    "sgmanet-loss-edge-w1-uieb800",
    "sgmanet-loss-edge-w2-uieb800",
    "sgmanet-loss-edge-w10-uieb800",
    "sgmanet-loss-ssim-full-uieb800",
    "sgmanet-loss-tv-w0001-uieb800",
    "sgmanet-loss-tv-w1-uieb800",
    "sgmanet-loss-lvw-w1-uieb800",
    "sgmanet-loss-uiqm-w1-uieb800",
    "sgmanet-loss-hvi-w1-uieb800",
    "sgmanet-loss-ssim-w1-uieb800",
}


# Ensure package root and local runner dir are on sys.path
_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from runner.project_config import accounts_by_id, get_project
from runner.job_builder import push_job, status_job, download_job_output, WORK_DIR


def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def get_jobs_status():
    # 1. Try server first
    try:
        r = requests.get(f"{SERVER_URL}/api/projects/{PROJECT_NAME}/jobs_status?force_refresh=true", timeout=15)
        if r.status_code == 200:
            return r.json().get("jobs", [])
    except Exception:
        pass

    # 2. Direct fallback via runner modules
    try:
        accounts = accounts_by_id()
        project = get_project(PROJECT_NAME)
        if not project:
            return None
        jobs = []
        for j in project.get("jobs", []):
            acc_id = j.get("account_id")
            acc = accounts.get(acc_id)
            if not acc:
                continue
            st_res = status_job(acc, j, force_refresh=True)
            st_out = (st_res.stdout + st_res.stderr).upper()
            if "RUNNING" in st_out:
                st = "RUNNING"
            elif "COMPLETE" in st_out:
                st = "COMPLETE"
            elif "404" in st_out or "NOT FOUND" in st_out:
                st = "IDLE"
            elif "ERROR" in st_out:
                st = "ERROR"
            elif "QUEUED" in st_out:
                st = "QUEUED"
            else:
                st = "IDLE"
            jobs.append({
                "name": j.get("name"),
                "account_id": acc_id,
                "status": st,
            })
        return jobs
    except Exception as e:
        log(f"Error checking jobs status: {e}")
    return None


def run_job(job_name: str):
    log(f"--> Triggering job dispatch: {job_name}")
    try:
        r = requests.post(
            f"{SERVER_URL}/api/jobs/run",
            json={"project_name": PROJECT_NAME, "job_names": [job_name]},
            timeout=30,
        )
        if r.status_code == 200:
            res = r.json()
            log(f"    Dispatch response via HTTP: {res}")
            return res.get("success", False)
    except Exception:
        pass

    # Direct fallback
    try:
        accounts = accounts_by_id()
        project = get_project(PROJECT_NAME)
        for j in project.get("jobs", []):
            if j.get("name") == job_name:
                acc = accounts.get(j.get("account_id"))
                job_dir = WORK_DIR / PROJECT_NAME / f"uwir-{job_name}"
                res = push_job(acc, job_dir)
                log(f"    Direct push response: {res.stdout or res.stderr}")
                return res.returncode == 0
    except Exception as e:
        log(f"    Direct push failed: {e}")
    return False


def download_job(job_name: str):
    log(f"<-- Downloading output for completed job: {job_name}")
    try:
        r = requests.post(
            f"{SERVER_URL}/api/jobs/download",
            json={"project_name": PROJECT_NAME, "job_names": [job_name]},
            timeout=60,
        )
        if r.status_code == 200:
            return True
    except Exception:
        pass

    # Direct fallback
    try:
        accounts = accounts_by_id()
        project = get_project(PROJECT_NAME)
        for j in project.get("jobs", []):
            if j.get("name") == job_name:
                acc = accounts.get(j.get("account_id"))
                out_dir, res = download_job_output(PROJECT_NAME, acc, j)
                log(f"    Direct download finished to {out_dir}: {res.stdout or res.stderr}")
                return True
    except Exception as e:
        log(f"    Direct download failed: {e}")
    return False


def main():
    log(f"Starting Auto-Dispatcher for project: {PROJECT_NAME}")
    log(f"Target pending jobs: {list(PENDING_DISPATCH.keys())}")

    while True:
        jobs = get_jobs_status()
        if not jobs:
            time.sleep(POLL_INTERVAL)
            continue

        account_running_counts = {"account2": 0, "account3": 0}
        statuses_summary = {}

        for j in jobs:
            name = j.get("name")
            acc_id = j.get("account_id")
            st = (j.get("status") or "UNKNOWN").upper()
            statuses_summary[name] = st

            if st == "RUNNING" or st == "QUEUED":
                account_running_counts[acc_id] = account_running_counts.get(acc_id, 0) + 1
            elif st == "COMPLETE":
                if name not in DOWNLOADED_JOBS:
                    log(f"Job {name} has COMPLETED! Initiating output download...")
                    download_job(name)
                    DOWNLOADED_JOBS.add(name)

        log(
            f"Status Check | Acc2 (TThanh13): {account_running_counts.get('account2', 0)}/2 running | "
            f"Acc3 (RaH1111): {account_running_counts.get('account3', 0)}/2 running | "
            f"Summary: {statuses_summary}"
        )

        # Check if we can dispatch SSIM on account2
        ssim_info = PENDING_DISPATCH["sgmanet-loss-ssim-uieb800"]
        if not ssim_info["dispatched"]:
            curr_st = statuses_summary.get("sgmanet-loss-ssim-uieb800", "IDLE")
            if curr_st in ["RUNNING", "COMPLETE"]:
                ssim_info["dispatched"] = True
            elif account_running_counts.get("account2", 0) < 2:
                log("Slot available on Account 2 (TThanh13)! Dispatching sgmanet-loss-ssim-uieb800...")
                if run_job("sgmanet-loss-ssim-uieb800"):
                    ssim_info["dispatched"] = True
                    account_running_counts["account2"] = account_running_counts.get("account2", 0) + 1

        # Check if we can dispatch HVI on account3
        hvi_info = PENDING_DISPATCH["sgmanet-loss-hvi-uieb800"]
        if not hvi_info["dispatched"]:
            curr_st = statuses_summary.get("sgmanet-loss-hvi-uieb800", "IDLE")
            if curr_st in ["RUNNING", "COMPLETE"]:
                hvi_info["dispatched"] = True
            elif account_running_counts.get("account3", 0) < 2:
                log("Slot available on Account 3 (RaH1111)! Dispatching sgmanet-loss-hvi-uieb800...")
                if run_job("sgmanet-loss-hvi-uieb800"):
                    hvi_info["dispatched"] = True
                    account_running_counts["account3"] = account_running_counts.get("account3", 0) + 1

        # Check if all done
        all_dispatched = all(v["dispatched"] for v in PENDING_DISPATCH.values())
        all_complete = all(st == "COMPLETE" for st in statuses_summary.values())
        if all_dispatched and all_complete:
            log("All ablation jobs completed and downloaded! Auto-Dispatcher exiting.")
            break

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
