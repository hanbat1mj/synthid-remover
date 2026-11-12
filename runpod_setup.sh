#!/usr/bin/env bash
# SynthID Remover — RunPod one-shot setup + launch.
#
# Paste into RunPod's Web Terminal (or JupyterLab terminal), or run via SSH.
# It clones the repo, installs deps, then launches the Gradio UI on 0.0.0.0:7860
# (RunPod proxies that port to an HTTPS URL).
set -euo pipefail

REPO_URL="${REPO_URL:-https://huggingface.co/spaces/dennny123/synth-id-remover}"
WORKDIR="${WORKDIR:-/workspace/synthid-remover}"
# Low-memory mode: leave unset to auto-detect (24GB GPU -> resident; <24GB -> swap).
# Force with: export SYNTHID_LOW_MEMORY=1
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

echo "==> [1/3] Cloning repo"
if [ -d "$WORKDIR/.git" ]; then
    (cd "$WORKDIR" && git pull --ff-only || true)
else
    mkdir -p "$(dirname "$WORKDIR")"
    git clone "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

echo "==> [2/3] Installing Python dependencies"
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "==> [3/3] Launching (first run clones ComfyUI + downloads 28 GiB)"
python app.py
