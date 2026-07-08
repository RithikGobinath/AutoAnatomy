#!/usr/bin/env bash
# Sets up the AutoAnatomy dev environment: isolated .venv, CUDA-matched torch,
# trimmed dependency set, and the real craniofacial_structures + skull-crop
# model weights. Safe to re-run.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

echo "== AutoAnatomy setup =="
df -h "$root" | tail -1

if [ ! -d ".venv" ]; then
    echo "Creating .venv ..."
    python3 -m venv .venv
fi

PY=".venv/bin/python"

echo "Upgrading pip ..."
"$PY" -m pip install --upgrade pip --quiet

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected, installing CUDA-enabled torch (cu128) ..."
    "$PY" -m pip install --force-reinstall --no-deps "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128
else
    echo "No NVIDIA GPU detected -- installing CPU-only torch."
fi

echo "Installing autoanatomy + dependencies (editable) ..."
"$PY" -m pip install -e .

# Re-pin the CUDA build: nnunetv2's own torchvision pin can silently replace it
# with a CPU wheel during the -e . install above.
if command -v nvidia-smi >/dev/null 2>&1; then
    "$PY" -m pip install --force-reinstall --no-deps "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128
fi

"$PY" -m pip cache purge >/dev/null || true

cuda_ok=$("$PY" -c "import torch; print(torch.cuda.is_available())")
if [ "$cuda_ok" = "True" ]; then
    echo "CUDA is working."
else
    echo "CUDA not available -- AutoAnatomy will fall back to --device cpu (slower)."
fi

echo "Downloading model weights (craniofacial_structures + skull-crop model) ..."
"$PY" -m autoanatomy.cli.main download-weights

df -h "$root" | tail -1
echo "Setup complete. Run '.venv/bin/autoanatomy-gui' to launch the TUI."
