# ==============================================================================
# train_liteenhancenet.ps1
# Train and benchmark LiteEnhanceNet (3ch & 5ch) on EUVP dataset using RTX 5070 GPU
# ==============================================================================

$ErrorActionPreference = "Stop"

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUTF8 = "1"

$PYTHON = "C:\ProgramData\anaconda3\envs\hungpt19\python.exe"
$PROJECT_DIR = Resolve-Path "$PSScriptRoot\..\.."

Set-Location $PROJECT_DIR

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " Training & Benchmarking LiteEnhanceNet (3ch & 5ch) on EUVP" -ForegroundColor Cyan
Write-Host " Python: $PYTHON" -ForegroundColor Cyan
Write-Host " Project Root: $PROJECT_DIR" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# Output directories
$CKPT_DIR = "./checkpoints/ablation_liteenhancenet"
$RESULTS_DIR = "./results/ablation_liteenhancenet"

if (!(Test-Path -Path $CKPT_DIR)) { New-Item -ItemType Directory -Path $CKPT_DIR -Force | Out-Null }
if (!(Test-Path -Path $RESULTS_DIR)) { New-Item -ItemType Directory -Path $RESULTS_DIR -Force | Out-Null }

# Run automated training and evaluation across both 3ch and 5ch variants
& $PYTHON -m scripts.experiments.ablation_euvp `
    --variants liteenhancenet_3ch liteenhancenet_5ch `
    --data_train_euvp ./datasets/EUVP `
    --checkpoint_dir $CKPT_DIR `
    --val_folder $RESULTS_DIR `
    --prior_method udcp `
    --nEpochs 50 `
    --batchSize 16 `
    --cropSize 256 `
    --lr 1e-4 `
    --amp `
    --L1_weight 1.0 `
    --perceptual_weight 1.0 `
    --SSIM_weight 0.1 `
    --seeds 0

Write-Host "================================================================" -ForegroundColor Green
Write-Host " Training & Evaluation completed!" -ForegroundColor Green
Write-Host " Results saved to: $RESULTS_DIR/ablation_train_multi_run_results.json" -ForegroundColor Green
Write-Host " Checkpoints saved to: $CKPT_DIR" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
