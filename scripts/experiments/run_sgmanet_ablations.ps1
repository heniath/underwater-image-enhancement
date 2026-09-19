<#
.SYNOPSIS
    Run SGMA-Net Loss Ablations (Edge Loss w1/w2/w10, TV Loss w1/w0.001/w0.01, SSIM Full) on Windows PowerShell.
.EXAMPLE
    # Run all ablations:
    .\scripts\experiments\run_sgmanet_ablations.ps1

    # Run only TV Loss ablations:
    .\scripts\experiments\run_sgmanet_ablations.ps1 -Filter "tv"

    # Run only Edge Loss ablations:
    .\scripts\experiments\run_sgmanet_ablations.ps1 -Filter "edge"
#>

param (
    [string]$Filter = "",
    [string]$ModelVariant = "sgmanet_5ch",
    [int]$Epochs = 100,
    [int]$BatchSize = 16,
    [int]$CropSize = 256,
    [string]$Lr = "1e-4",
    [string]$Amp = "True",
    [string]$InMemory = "False",
    [int]$ValInterval = 1,
    [string]$DataUieb = "datasets/UIEB",
    [string]$DataEuvp = "datasets/EUVP",
    [int]$UiebLimit = 800,
    [int]$NumGpus = 1
)

$ErrorActionPreference = "Continue"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "src"

$ProjectDir = (Get-Item $PSScriptRoot\..\..).FullName
Set-Location $ProjectDir

if ([string]::IsNullOrEmpty($DataUieb) -or $DataUieb -eq "datasets/UIEB") {
    if (Test-Path "$ProjectDir\..\Dataset\UIEB") {
        $DataUieb = "$ProjectDir\..\Dataset\UIEB"
    } else {
        $DataUieb = "$ProjectDir\datasets\UIEB"
    }
}

if ([string]::IsNullOrEmpty($DataEuvp) -or $DataEuvp -eq "datasets/EUVP") {
    if (Test-Path "$ProjectDir\..\Dataset\EUVP") {
        $DataEuvp = "$ProjectDir\..\Dataset\EUVP"
    } else {
        $DataEuvp = "$ProjectDir\datasets\EUVP"
    }
}

New-Item -ItemType Directory -Force -Path "logs", "checkpoints", "results" | Out-Null

$AllRuns = @(
    @{ Name = "sgmanet_5ch_uieb_edge_w1";    Flags = @("--use_l1", "1", "--use_perc", "1", "--use_edge", "1", "--edge_weight", "1.0") },
    @{ Name = "sgmanet_5ch_uieb_edge_w2";    Flags = @("--use_l1", "1", "--use_perc", "1", "--use_edge", "1", "--edge_weight", "2.0") },
    @{ Name = "sgmanet_5ch_uieb_edge_w10";   Flags = @("--use_l1", "1", "--use_perc", "1", "--use_edge", "1", "--edge_weight", "10.0") },
    @{ Name = "sgmanet_5ch_uieb_tv_w1";      Flags = @("--use_l1", "1", "--use_perc", "1", "--use_tv", "1", "--tv_weight", "1.0") },
    @{ Name = "sgmanet_5ch_uieb_tv_w0001";   Flags = @("--use_l1", "1", "--use_perc", "1", "--use_tv", "1", "--tv_weight", "0.001") },
    @{ Name = "sgmanet_5ch_uieb_tv_w001";    Flags = @("--use_l1", "1", "--use_perc", "1", "--use_tv", "1", "--tv_weight", "0.01") },
    @{ Name = "sgmanet_5ch_uieb_ssim_full";  Flags = @("--use_l1", "1", "--use_perc", "1", "--use_ssim", "1", "--SSIM_weight", "0.1", "--early_stop_patience", "100") }
)

$Runs = @()
foreach ($r in $AllRuns) {
    if ([string]::IsNullOrEmpty($Filter) -or $r.Name -like "*$Filter*") {
        $Runs += $r
    }
}

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " Starting SGMA-Net Loss Ablation Suite ($($Runs.Count) runs)" -ForegroundColor Cyan
Write-Host " Filter     : $(if ($Filter) { $Filter } else { 'ALL' })"
Write-Host " Model      : $ModelVariant"
Write-Host " Epochs     : $Epochs"
Write-Host " Batch Size : $BatchSize"
Write-Host " Device     : Windows Native CUDA (RTX)"
Write-Host " Start at   : $(Get-Date)"
Write-Host "=================================================================" -ForegroundColor Cyan

$Idx = 0
foreach ($run in $Runs) {
    $Idx++
    $Timestamp = (Get-Date -Format "yyyyMMdd_HHmmss")
    $LogFile = "logs\$($run.Name)_$Timestamp.log"

    Write-Host "`n-----------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host " [Run $Idx/$($Runs.Count)] $($run.Name)" -ForegroundColor Green
    Write-Host " Flags: $($run.Flags -join ' ')"
    Write-Host " Log  : $LogFile"
    Write-Host "-----------------------------------------------------------------" -ForegroundColor Yellow

    $ArgsList = @(
        "-m", "uwir.cli.train",
        "--model", $ModelVariant,
        "--dataset", "uieb",
        "--data_train_uieb", $DataUieb,
        "--uieb_limit", $UiebLimit,
        "--run_name", $run.Name,
        "--nEpochs", $Epochs,
        "--batchSize", $BatchSize,
        "--cropSize", $CropSize,
        "--lr", $Lr,
        "--amp", $Amp,
        "--in_memory", $InMemory,
        "--val_interval", $ValInterval,
        "--num_gpus", $NumGpus
    ) + $run.Flags

    $Sw = [System.Diagnostics.Stopwatch]::StartNew()
    python $ArgsList
    $Sw.Stop()

    if ($LASTEXITCODE -ne 0) {
        Write-Host ">> [Error] Run $Idx failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    Write-Host ">> [Run $Idx] Completed in $([math]::Round($Sw.Elapsed.TotalSeconds, 1))s" -ForegroundColor Green

    # Clear memory
    python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>$null
}

Write-Host "`n=================================================================" -ForegroundColor Cyan
Write-Host " Evaluating checkpoints on UIEB-90 and EUVP benchmarks" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

python -m uwir.cli.evaluate `
    --checkpoint_dir "./checkpoints" `
    --eval_benchmark "uieb+euvp" `
    --data_train_uieb $DataUieb `
    --data_train_euvp $DataEuvp `
    --val_folder "./results/eval_ablations"

Write-Host "`n [ALL FINISHED] All runs and benchmarks completed!" -ForegroundColor Green
