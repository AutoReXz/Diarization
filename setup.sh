#!/usr/bin/env bash
# ============================================================
# setup.sh — One-shot environment setup for the diarization
#             pipeline on Linux / macOS.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ============================================================

set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR=".venv"

echo "=================================================="
echo "  Diarization Pipeline — Environment Setup"
echo "=================================================="

# ── 1. Check Python version ────────────────────────────
echo ""
echo "[1/5] Checking Python …"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON not found. Install Python 3.9–3.11."
    exit 1
fi
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Found $PYTHON ($PY_VER)"

# ── 2. Check ffmpeg ────────────────────────────────────
echo ""
echo "[2/5] Checking ffmpeg …"
if ! command -v ffmpeg &>/dev/null; then
    echo "  ⚠  ffmpeg not found — attempting to install …"
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
    elif command -v brew &>/dev/null; then
        brew install ffmpeg
    else
        echo "ERROR: Cannot auto-install ffmpeg. Install it manually."
        exit 1
    fi
fi
FFMPEG_VER=$(ffmpeg -version | head -1)
echo "  $FFMPEG_VER"

# ── 3. Create virtual environment ──────────────────────
echo ""
echo "[3/5] Creating virtual environment in $VENV_DIR …"
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv "$VENV_DIR"
    echo "  Created."
else
    echo "  Already exists, reusing."
fi
source "$VENV_DIR/bin/activate"
echo "  Activated: $(which python)"

# ── 4. Upgrade pip + install deps ──────────────────────
echo ""
echo "[4/5] Installing dependencies …"
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q

# ── 5. Verify imports ─────────────────────────────────
echo ""
echo "[5/5] Verifying key imports …"
python -c "
import torch, whisperx, pyannote.audio
print(f'  torch      : {torch.__version__}  (CUDA: {torch.cuda.is_available()})')
print(f'  whisperx   : OK')
print(f'  pyannote   : {pyannote.audio.__version__}')
"

echo ""
echo "=================================================="
echo "  Setup complete!"
echo ""
echo "  Activate the environment:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Run the pipeline:"
echo "    python pipeline.py your_audio.wav --hf-token hf_XXXXX"
echo ""
echo "  (Set HF_TOKEN env var to skip --hf-token each time)"
echo "=================================================="
