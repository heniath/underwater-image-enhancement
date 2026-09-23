#!/usr/bin/env python3
import json
import sys
import zipfile
from pathlib import Path

runner_dir = Path(__file__).resolve().parent
if str(runner_dir) not in sys.path:
    sys.path.insert(0, str(runner_dir))

from runner.project_config import accounts_by_id, get_project
from runner.job_builder import download_job_output, kernel_id, status_job

TARGET_JOBS = [
    # Round 6
    "sgmanet-combo-lvw5-uiqm02-uieb800",
    "sgmanet-combo-lvw5-uiqm1-uieb800",
    "sgmanet-combo-lvw5-uiqm01-uieb800",
    # Round 7
    "sgmanet-base-harm-lvw2-uiqm02-uieb800",
    "sgmanet-base-f11-lvw2-uiqm02-uieb800",
    "sgmanet-base-gd-lvw2-uiqm02-uieb800",
]

def main():
    accounts = accounts_by_id()
    project = get_project("sgmanet_loss_ablations")
    if not project:
        print("Project not found!")
        sys.exit(1)

    jobs_by_name = {j["name"]: j for j in project["jobs"]}

    for name in TARGET_JOBS:
        job = jobs_by_name.get(name)
        if not job:
            continue
        acc = accounts.get(job["account_id"])
        st = status_job(acc, job, force_refresh=True)
        st_text = st.stdout.strip() or st.stderr.strip()
        is_complete = "COMPLETE" in st_text
        print(f"\n=======================================================")
        print(f"Job: {name} ({acc['username']}) -> Status: {st_text}")

        if is_complete:
            out_dir, res = download_job_output("sgmanet_loss_ablations", acc, job)
            print(f"  Download result code: {res.returncode}")
            # Check zip files to unpack
            for z in out_dir.glob("*.zip"):
                try:
                    with zipfile.ZipFile(z, 'r') as zip_ref:
                        zip_ref.extractall(out_dir)
                    print(f"  Extracted archive: {z.name}")
                except Exception as e:
                    print(f"  Error extracting {z.name}: {e}")

            # Check test_results_all.json
            results_files = list(out_dir.glob("**/test_results_all.json"))
            if results_files:
                rf = results_files[0]
                try:
                    with open(rf, 'r') as f:
                        data = json.load(f)
                    print(f"  >>> Found benchmark metrics:")
                    if "uieb" in data:
                        print(f"      UIEB-90 : PSNR={data['uieb'].get('psnr'):.4f} dB, SSIM={data['uieb'].get('ssim'):.4f}, CIEDE={data['uieb'].get('ciede2000'):.4f}, UIQM={data['uieb'].get('uiqm'):.4f}, UCIQE={data['uieb'].get('uciqe'):.4f}")
                    if "euvp" in data:
                        print(f"      EUVP-515: PSNR={data['euvp'].get('psnr'):.4f} dB, SSIM={data['euvp'].get('ssim'):.4f}, CIEDE={data['euvp'].get('ciede2000'):.4f}, UIQM={data['euvp'].get('uiqm'):.4f}")
                    if "overall" in data:
                        print(f"      Overall : PSNR={data['overall'].get('psnr'):.4f} dB, SSIM={data['overall'].get('ssim'):.4f}")
                except Exception as e:
                    print(f"  Error reading {rf}: {e}")
            else:
                print(f"  No test_results_all.json found yet in {out_dir}")

if __name__ == "__main__":
    main()
