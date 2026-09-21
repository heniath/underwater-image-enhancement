# ==============================================================================
# run_lightweight_tournament.ps1
# Automated Training & Benchmarking Suite for Lightweight UWIR Models
# ==============================================================================

param (
    [string]$Action = "compare",          # "train", "test", "profile", "compare", "tournament"
    [string]$Model = "fgdpa_3ch",         # e.g. fgdpa_3ch, mobileie_3ch, liteenhancenet_3ch, lsnet_3ch
    [string]$Dataset = "uieb",            # "uieb" or "euvp"
    [int]$Epochs = 50,
    [int]$BatchSize = 16,
    [switch]$Amp = $true
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUTF8 = "1"

$PYTHON = "C:\ProgramData\anaconda3\envs\hungpt19\python.exe"
$PROJECT_DIR = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $PROJECT_DIR

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " UWIR Lightweight Models Suite (RTX 5070 Optimized)" -ForegroundColor Cyan
Write-Host " Action:   $Action" -ForegroundColor Cyan
Write-Host " Model:    $Model" -ForegroundColor Cyan
Write-Host " Dataset:  $Dataset" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

switch ($Action.ToLower()) {
    "train" {
        $ampFlag = if ($Amp) { "--amp" } else { "--no_amp" }
        & $PYTHON run.py train -m $Model -d $Dataset -e $Epochs -b $BatchSize $ampFlag
    }
    "test" {
        & $PYTHON run.py test -m $Model -d $Dataset
    }
    "profile" {
        & $PYTHON run.py profile
    }
    "compare" {
        & $PYTHON run.py compare -d $Dataset
    }
    "tournament" {
        $models = @("fgdpa_3ch", "mobileie_3ch", "liteenhancenet_3ch", "lsnet_3ch")
        foreach ($m in $models) {
            Write-Host "`n>>> [TOURNAMENT] Training $m for $Epochs epochs..." -ForegroundColor Yellow
            & $PYTHON run.py train -m $m -d $Dataset -e $Epochs -b $BatchSize --amp
        }
        Write-Host "`n>>> [TOURNAMENT] Generating Leaderboard..." -ForegroundColor Green
        & $PYTHON run.py compare -d $Dataset
    }
    default {
        Write-Host "Unknown action: $Action. Choose from: train, test, profile, compare, tournament" -ForegroundColor Red
    }
}
