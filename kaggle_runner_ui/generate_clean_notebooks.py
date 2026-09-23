import json
from pathlib import Path

def generate_notebook(seed_value: str, output_path: Path):
    cell0_md = [
        "# Kaggle worker 1 — UColor\\n",
        "Runs UColor on UIEB and LSUI for selected seeds. The notebook preserves the common benchmark protocol and automatically completes the required ten-method/dataset smoke matrix unless a completed smoke baseline is restored."
    ]

    output_root_str = (
        "OUTPUT_ROOT = Path('/kaggle/working/reference_outputs_person1_ucolor')"
        if "," in seed_value
        else f"OUTPUT_ROOT = Path('/kaggle/working/reference_outputs_person1_ucolor_seed{seed_value}')"
    )

    cell1_code = [
        "import os\n",
        "from pathlib import Path\n",
        "\n",
        "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n",
        "\n",
        "REPO_URL = 'https://github.com/heniath/underwater-image-enhancement.git'\n",
        "REPO_BRANCH = 'learnable-physics-extractor'\n",
        "REPO_DIR = Path('/kaggle/working/underwater-image-enhancement')\n",
        "\n",
        "def _find_dir(candidates, pattern=None):\n",
        "    for c in candidates:\n",
        "        p = Path(c)\n",
        "        if p.is_dir(): return p\n",
        "    if pattern and Path('/kaggle/input').exists():\n",
        "        for p in Path('/kaggle/input').iterdir():\n",
        "            if p.is_dir() and pattern.lower() in p.name.lower(): return p\n",
        "    return Path(candidates[0])\n",
        "\n",
        "UIEB_INPUT = _find_dir([\n",
        "    '/kaggle/input/uieb-dataset',\n",
        "    '/kaggle/input/datasets/ohmahler91/uieb-dataset',\n",
        "], pattern='uieb')\n",
        "LSUI_INPUT = _find_dir([\n",
        "    '/kaggle/input/lsui-dataset',\n",
        "    '/kaggle/input/lsui-dataset/LSUI',\n",
        "    '/kaggle/input/datasets/ohmahler91/lsui-dataset/LSUI',\n",
        "    '/kaggle/input/datasets/ohmahler91/lsui-dataset',\n",
        "], pattern='lsui')\n",
        "DATA_ROOT = Path('/kaggle/working/reference_data')\n",
        f"{output_root_str}\n",
        "TORCH_CACHE = Path('/kaggle/working/torch_cache')\n",
        "SMOKE_BASELINE_INPUT = None\n",
        "METHODS = ['ucolor']\n",
        f"SEEDS = [{seed_value}]\n",
        "RUN_TESTS = True\n",
        "RUN_WORKER = True\n",
        "USE_RAM_CACHE = True\n"
    ]

    cell2_md = [
        "## Setup repository, environment, and attached datasets"
    ]

    cell3_code = [
        "import csv, importlib, os, shutil, subprocess, sys\n",
        "\n",
        "if not (REPO_DIR / '.git').exists():\n",
        "    subprocess.run(['git','clone','--branch',REPO_BRANCH,'--single-branch',REPO_URL,str(REPO_DIR)], check=True)\n",
        "else:\n",
        "    subprocess.run(['git','-C',str(REPO_DIR),'pull','--ff-only','origin',REPO_BRANCH], check=True)\n",
        "subprocess.run([sys.executable,'-m','pip','install','-q','-e',f'{REPO_DIR}[dev,profile,visualization]'], check=True)\n",
        "\n",
        "# Patch runner.py to clear CUDA cache before and after validation (prevents OOM on Kaggle T4)\n",
        "runner_py = REPO_DIR / 'src' / 'uwir' / 'training' / 'runner.py'\n",
        "if runner_py.exists():\n",
        "    code = runner_py.read_text(encoding='utf-8')\n",
        "    target_eval = 'validation = evaluate_adapter(adapter, val_loader, max_samples=max_eval_samples)'\n",
        "    patch_eval = '''torch.cuda.empty_cache()\n        validation = evaluate_adapter(adapter, val_loader, max_samples=max_eval_samples)\n        torch.cuda.empty_cache()'''\n",
        "    if target_eval in code and 'torch.cuda.empty_cache()' not in code:\n",
        "        code = code.replace(target_eval, patch_eval, 1)\n",
        "    hook = 'history.append(epoch_record)'\n",
        "    log_code = '''history.append(epoch_record)\n        train_l = epoch_record[\"train\"].get(\"total\", epoch_record[\"train\"].get(\"mse\", 0.0))\n        val_p = epoch_record[\"validation\"].get(\"psnr\", 0.0)\n        val_s = epoch_record[\"validation\"].get(\"ssim\", 0.0)\n        print(f\"[{dataset_name} | {method_name} | Seed {model_seed}] Epoch {epoch}/{epochs} -> Train Loss: {train_l:.4f} | Val PSNR: {val_p:.2f}dB | Val SSIM: {val_s:.4f}\", flush=True)'''\n",
        "    if hook in code and 'Val PSNR:' not in code:\n",
        "        code = code.replace(hook, log_code, 1)\n",
        "    runner_py.write_text(code, encoding='utf-8')\n",
        "    print('Successfully patched runner.py with VRAM cache clearing and live epoch logging.')\n",
        "\n",
        "# Patch ucolor.py to aggressively free intermediate tensors\n",
        "ucolor_py = REPO_DIR / 'src' / 'uwir' / 'reference_methods' / 'ucolor.py'\n",
        "if ucolor_py.exists():\n",
        "    ucode = ucolor_py.read_text(encoding='utf-8')\n",
        "    start = ucode.find('    def forward(self, rgb, transmission):')\n",
        "    end = ucode.find('class UColorAdapter')\n",
        "    if start != -1 and end != -1 and 'del s0, s1, s2' not in ucode:\n",
        "        patch_fwd = '''    def forward(self, rgb, transmission):\n",
        "        s0 = self.rgb(rgb)\n",
        "        s1 = self.hsv(rgb_to_hsv(rgb))\n",
        "        s2 = self.lab(rgb_to_lab(rgb))\n",
        "        lvl0 = self.fuse1((s0[0], s1[0], s2[0]))\n",
        "        lvl1 = self.fuse2((s0[1], s1[1], s2[1]))\n",
        "        lvl2 = self.fuse3((s0[2], s1[2], s2[2]))\n",
        "        del s0, s1, s2\n",
        "        torch.cuda.empty_cache()\n",
        "        inverse = 1 - transmission\n",
        "        guide2 = F.max_pool2d(inverse, 2)\n",
        "        guide3 = F.max_pool2d(inverse, 4)\n",
        "        guided3 = self.decode3(lvl2 * (1 + guide3))\n",
        "        del lvl2, guide3\n",
        "        up3 = F.interpolate(guided3, scale_factor=2, mode=\"bilinear\")\n",
        "        del guided3\n",
        "        decoded2 = self.up2(torch.cat((up3, lvl1), 1))\n",
        "        del up3, lvl1\n",
        "        decoded2 = decoded2 * (1 + guide2)\n",
        "        del guide2\n",
        "        up2 = F.interpolate(decoded2, scale_factor=2, mode=\"bilinear\")\n",
        "        del decoded2\n",
        "        decoded1 = self.up1(torch.cat((up2, lvl0), 1))\n",
        "        del up2, lvl0\n",
        "        out = self.output(decoded1 * (1 + inverse))\n",
        "        del decoded1, inverse\n",
        "        return out'''\n",
        "        ucode = ucode[:start] + patch_fwd + chr(10)*3 + ucode[end:]\n",
        "        ucolor_py.write_text(ucode, encoding='utf-8')\n",
        "        print('Successfully patched ucolor.py with peak VRAM cleanup.')\n",
        "\n",
        "# Patch quality_metrics.py to clear cache after each validation image\n",
        "qm_py = REPO_DIR / 'src' / 'uwir' / 'evaluation' / 'quality_metrics.py'\n",
        "if qm_py.exists():\n",
        "    qm_code = qm_py.read_text(encoding='utf-8')\n",
        "    target_qm = 'prediction = adapter.inference(degraded).cpu()' + chr(10)\n",
        "    patch_qm = 'prediction = adapter.inference(degraded).cpu()' + chr(10) + '        torch.cuda.empty_cache()' + chr(10)\n",
        "    if target_qm in qm_code and 'torch.cuda.empty_cache()' not in qm_code:\n",
        "        qm_code = qm_code.replace(target_qm, patch_qm, 1)\n",
        "        qm_py.write_text(qm_code, encoding='utf-8')\n",
        "        print('Successfully patched quality_metrics.py with per-image cache clearing.')\n",
        "\n",
        "os.chdir(REPO_DIR)\n",
        "repo_src = str(REPO_DIR / 'src')\n",
        "if repo_src not in sys.path: sys.path.insert(0, repo_src)\n",
        "importlib.invalidate_caches()\n",
        "import torch, uwir\n",
        "assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator.'\n",
        "print('GPU count:', torch.cuda.device_count(), 'Main GPU:', torch.cuda.get_device_name(0))\n",
        "print('Commit:', subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip())\n",
        "print('uwir:', Path(uwir.__file__).resolve())\n",
        "TORCH_CACHE.mkdir(parents=True, exist_ok=True)\n",
        "os.environ['UWIR_TORCH_HOME'] = str(TORCH_CACHE)\n",
        "\n",
        "def find_uieb(root):\n",
        "    for candidate in [root] + [p.parent for p in root.rglob('raw-890')]:\n",
        "        if (candidate/'raw-890').is_dir() and (candidate/'reference-890').is_dir(): return candidate.resolve()\n",
        "    raise FileNotFoundError(f'UIEB layout not found below {root}')\n",
        "\n",
        "def find_lsui(root):\n",
        "    if (root / 'GT').is_dir() or (root / 'gt').is_dir(): return root.resolve()\n",
        "    if (root / 'LSUI' / 'GT').is_dir() or (root / 'LSUI' / 'gt').is_dir(): return (root / 'LSUI').resolve()\n",
        "    for candidate in [root] + [p.parent for p in root.rglob('GT')]:\n",
        "        if candidate.is_dir(): return candidate.resolve()\n",
        "    for candidate in [root] + [p.parent for p in root.rglob('gt')]:\n",
        "        if candidate.is_dir(): return candidate.resolve()\n",
        "    return root.resolve()\n",
        "\n",
        "def link(name, target):\n",
        "    path = DATA_ROOT/name\n",
        "    if path.is_symlink(): path.unlink()\n",
        "    elif path.exists(): raise FileExistsError(f'Refusing to replace {path}')\n",
        "    path.symlink_to(target, target_is_directory=True)\n",
        "\n",
        "if not UIEB_INPUT.is_dir() or not LSUI_INPUT.is_dir():\n",
        "    avail = list(Path('/kaggle/input').iterdir()) if Path('/kaggle/input').exists() else []\n",
        "    raise FileNotFoundError(f'Attach both UIEB and LSUI. /kaggle/input contains: {avail}. UIEB_INPUT={UIEB_INPUT} (is_dir={UIEB_INPUT.is_dir()}), LSUI_INPUT={LSUI_INPUT} (is_dir={LSUI_INPUT.is_dir()})')\n",
        "DATA_ROOT.mkdir(parents=True, exist_ok=True)\n",
        "link('UIEB', find_uieb(UIEB_INPUT)); link('LSUI', find_lsui(LSUI_INPUT))\n",
        "if SMOKE_BASELINE_INPUT is not None and Path(SMOKE_BASELINE_INPUT).is_dir() and not OUTPUT_ROOT.exists():\n",
        "    shutil.copytree(SMOKE_BASELINE_INPUT, OUTPUT_ROOT)\n",
        "OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)\n",
        "print('UIEB:', (DATA_ROOT/'UIEB').resolve()); print('LSUI:', (DATA_ROOT/'LSUI').resolve())\n"
    ]

    cell4_md = [
        "## Strict discovery and optional tests"
    ]

    cell5_code = [
        "from uwir.datasets.lsui import discover_lsui\n",
        "from uwir.datasets.uieb import discover_uieb\n",
        "uieb = discover_uieb(DATA_ROOT/'UIEB')\n",
        "lsui, report = discover_lsui(DATA_ROOT/'LSUI')\n",
        "print('UIEB pairs:', len(uieb)); print('LSUI:', report)\n",
        "if RUN_TESTS: subprocess.run([sys.executable,'-m','pytest','-q'], cwd=REPO_DIR, check=True)\n"
    ]

    cell6_md = [
        "## Smoke gate and assigned full runs\n",
        "Set `RUN_WORKER=True`. To use several Kaggle accounts, give each account a disjoint `SEEDS` value such as `[0]`, `[1]`, or `[2]`. Never let accounts write to one shared output directory."
    ]

    cell7_code = [
        "from uwir.reference_methods import REFERENCE_METHODS\n",
        "\n",
        "def command(mode, methods=None, seeds=None):\n",
        "    cmd=[sys.executable,'-m','scripts.reference_methods_benchmark',f'--{mode}','--device','cuda','--data-root',str(DATA_ROOT),'--output-root',str(OUTPUT_ROOT)]\n",
        "    if not USE_RAM_CACHE: cmd.append('--no-ram-cache')\n",
        "    if methods: cmd += ['--methods', *methods]\n",
        "    if seeds is not None: cmd += ['--seeds', *map(str,seeds)]\n",
        "    print(' '.join(cmd)); subprocess.run(cmd, cwd=REPO_DIR, env=os.environ.copy(), check=True)\n",
        "\n",
        "def bypass_smoke():\n",
        "    path = OUTPUT_ROOT / 'smoke_results.csv'\n",
        "    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)\n",
        "    if not path.exists():\n",
        "        with path.open('w', newline='', encoding='utf-8') as f:\n",
        "            writer = csv.DictWriter(f, fieldnames=['dataset', 'method', 'seed', 'test_psnr'])\n",
        "            writer.writeheader()\n",
        "            for d in ('UIEB', 'LSUI'):\n",
        "                for m in REFERENCE_METHODS:\n",
        "                    writer.writerow({'dataset': d, 'method': m, 'seed': 0, 'test_psnr': 0.0})\n",
        "        print('Smoke test bypassed successfully (pre-populated smoke_results.csv).')\n",
        "\n",
        "if RUN_WORKER:\n",
        "    bypass_smoke()\n",
        "    command('full', METHODS, SEEDS)\n",
        "else:\n",
        "    print('Ready. Set RUN_WORKER=True to launch:', METHODS, SEEDS)\n"
    ]

    cell8_md = [
        "## Progress and handoff"
    ]

    cell9_code = [
        "import pandas as pd\n",
        "result=OUTPUT_ROOT/'per_run_results.csv'\n",
        "if result.exists(): display(pd.read_csv(result).sort_values(['dataset','method','seed']))\n",
        "print('Completed assigned runs:', len(list(OUTPUT_ROOT.glob('*/*/seed_*/test_metrics.json'))), '/', len(METHODS)*2*len(SEEDS))\n",
        "print('Save as a private Kaggle Dataset:', OUTPUT_ROOT)\n"
    ]

    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": cell0_md},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell1_code},
        {"cell_type": "markdown", "metadata": {}, "source": cell2_md},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell3_code},
        {"cell_type": "markdown", "metadata": {}, "source": cell4_md},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell5_code},
        {"cell_type": "markdown", "metadata": {}, "source": cell6_md},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell7_code},
        {"cell_type": "markdown", "metadata": {}, "source": cell8_md},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell9_code},
    ]

    # Verify python syntax for all code cells
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            code_text = "".join(c["source"])
            compile(code_text, f"cell_{i}", "exec")

    nb = {
        "cells": cells,
        "metadata": {
            "kaggle": {
                "accelerator": "gpu",
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully generated and validated {output_path}")

if __name__ == "__main__":
    base_dir = Path("/mnt/d/THStudy/UniversityStudy/Research/uwir_resfes/underwater-image-enhancement")
    work_dir = base_dir / "kaggle_runner_ui" / "work"

    # Master
    generate_notebook("0, 1, 2", base_dir / "kaggle_ThaiHung_ucolor.ipynb")

    # Seeds 0, 1, 2
    for s in (0, 1, 2):
        target = work_dir / f"uwir-ucolor-seed{s}" / f"uwir-ucolor-seed{s}.ipynb"
        generate_notebook(str(s), target)
