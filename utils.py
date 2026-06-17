"""
utils.py — Utility functions for the diarization pipeline.

Handles:
  • Audio preprocessing (ffmpeg conversion to 16 kHz mono WAV)
  • Device detection (CUDA / MPS / CPU)
  • Output formatting (plain text + JSON)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any


# ── Device helpers ──────────────────────────────────────────
def get_device() -> str:
    """Return the best available PyTorch device string."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon MPS — supported by recent PyTorch but still
    # hit-or-miss with some ops; fall back to CPU if it errors.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_compute_type(device: str) -> str:
    """Pick a reasonable compute type for the given device."""
    if device == "cuda":
        return "float16"
    return "float32"   # CPU / MPS safer with float32


# ── Audio preprocessing ────────────────────────────────────
def ensure_ffmpeg() -> str:
    """Return path to ffmpeg binary or raise an informative error."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise EnvironmentError(
            "ffmpeg not found on PATH.  Install it first:\n"
            "  • Ubuntu/Debian : sudo apt install ffmpeg\n"
            "  • macOS (brew)  : brew install ffmpeg\n"
            "  • Windows       : choco install ffmpeg   OR   winget install ffmpeg\n"
        )
    return path


def preprocess_audio(input_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """
    Convert *any* audio/video file to 16 kHz mono WAV via ffmpeg.

    Parameters
    ----------
    input_path : path to the source audio / video file.
    output_dir : directory for the converted file (default: same dir as input).

    Returns
    -------
    Path to the converted WAV file.
    """
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{input_path.stem}_16k.wav"

    # Skip if already converted
    if out_path.exists() and out_path.stat().st_mtime >= input_path.stat().st_mtime:
        print(f"[utils] Reusing existing preprocessed file: {out_path}")
        return out_path

    ffmpeg = ensure_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-ac", "1",               # mono
        "-ar", "16000",           # 16 kHz
        "-sample_fmt", "s16",     # 16-bit signed int
        "-f", "wav",
        str(out_path),
    ]
    print(f"[utils] Converting audio → 16 kHz mono WAV …")
    print(f"        {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    print(f"[utils] Saved preprocessed audio: {out_path}")
    return out_path


# ── Timestamp formatting ───────────────────────────────────
def fmt_ts(seconds: float) -> str:
    """Format seconds → HH:MM:SS.mmm"""
    td = timedelta(seconds=seconds)
    total_secs = int(td.total_seconds())
    hrs, remainder = divmod(total_secs, 3600)
    mins, secs = divmod(remainder, 60)
    millis = int((seconds - total_secs) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


# ── Output formatters ──────────────────────────────────────
def segments_to_text(segments: list[dict[str, Any]]) -> str:
    """
    Convert diarised segments to readable plain-text transcript.

    Expected segment dict keys:
      speaker, start, end, text
    """
    lines: list[str] = []
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        start = fmt_ts(seg["start"])
        end = fmt_ts(seg["end"])
        text = seg["text"].strip()
        lines.append(f"[{start} → {end}]  {speaker}:  {text}")
    return "\n".join(lines)


def segments_to_json(segments: list[dict[str, Any]], pretty: bool = True) -> str:
    """Serialise diarised segments to JSON."""
    indent = 2 if pretty else None
    return json.dumps(segments, indent=indent, ensure_ascii=False)


def save_outputs(
    segments: list[dict[str, Any]],
    base_path: str | Path,
) -> tuple[Path, Path]:
    """
    Write both .txt and .json outputs next to (or based on) *base_path*.

    Returns (txt_path, json_path).
    """
    base = Path(base_path)
    txt_path = base.with_suffix(".txt")
    json_path = base.with_suffix(".json")

    txt_path.write_text(segments_to_text(segments), encoding="utf-8")
    json_path.write_text(segments_to_json(segments), encoding="utf-8")

    return txt_path, json_path
