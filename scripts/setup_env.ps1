#Requires -Version 5.1
# Sets up the AutoAnatomy dev environment: isolated .venv, CUDA-matched torch,
# trimmed dependency set, and the real craniofacial_structures + skull-crop
# model weights. Safe to re-run.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Get-FreeDiskGB {
    $drive = (Get-PSDrive -Name ($root.Substring(0,1)))
    return [math]::Round($drive.Free / 1GB, 1)
}

Write-Host "== AutoAnatomy setup ==" -ForegroundColor Cyan
Write-Host "Free disk before setup: $(Get-FreeDiskGB) GB"

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv ..."
    python -m venv .venv
}

$py = ".\.venv\Scripts\python.exe"

Write-Host "Upgrading pip ..."
& $py -m pip install --upgrade pip --quiet

# Detect GPU compute capability to pick a CUDA build that actually supports the
# card (e.g. Blackwell needs cu128+). Falls back to CPU-only torch if no NVIDIA
# GPU is found or the CUDA install fails -- the engine supports --device cpu.
$hasNvidia = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)

if ($hasNvidia) {
    Write-Host "NVIDIA GPU detected, installing CUDA-enabled torch (cu128) ..." -ForegroundColor Cyan
    & $py -m pip install --force-reinstall --no-deps "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "No NVIDIA GPU detected -- installing CPU-only torch." -ForegroundColor Yellow
}

Write-Host "Installing autoanatomy + dependencies (editable) ..."
& $py -m pip install -e .

# Re-pin the CUDA build: nnunetv2's own torchvision pin can silently replace it
# with a CPU wheel during the -e . install above.
if ($hasNvidia) {
    & $py -m pip install --force-reinstall --no-deps "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128
}

& $py -m pip cache purge | Out-Null

$cudaOk = & $py -c "import torch; print(torch.cuda.is_available())"
if ($cudaOk -eq "True") {
    Write-Host "CUDA is working." -ForegroundColor Green
} else {
    Write-Host "CUDA not available -- AutoAnatomy will fall back to --device cpu (slower)." -ForegroundColor Yellow
}

Write-Host "Downloading model weights (craniofacial_structures + skull-crop model) ..."
& $py -m autoanatomy.cli.main download-weights

Write-Host "Free disk after setup: $(Get-FreeDiskGB) GB"
Write-Host "Setup complete. Run '.venv\Scripts\autoanatomy-gui' to launch the TUI." -ForegroundColor Green
