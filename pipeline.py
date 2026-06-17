#!/usr/bin/env python3
"""
pipeline.py — Full speech-to-text + speaker diarization pipeline.

Models used (100 % open-source, self-hosted):
  • WhisperX  — ASR, word-level timestamps, forced alignment
  • pyannote/speaker-diarization-community-1  — speaker diarization (HuggingFace)

Usage
-----
    python pipeline.py audio.wav                        # defaults
    python pipeline.py audio.mp3 --language id          # Indonesian
    python pipeline.py meeting.m4a --model large-v3 --num-speakers 3
    python pipeline.py call.wav --hf-token hf_xxxxx     # pass HF token explicitly

The script produces two output files next to the input:
  • <name>_diarized.txt   — human-readable transcript
  • <name>_diarized.json  — machine-readable JSON
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

# Suppress noisy warnings from libraries
warnings.filterwarnings("ignore", message=".*torchaudio.*backend.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Compatibility shim: transformers >= 4.42 removed Pipeline from the top-level
# API; whisperx and pyannote both do `from transformers import Pipeline`.
import transformers as _transformers
if not hasattr(_transformers, "Pipeline"):
    try:
        from transformers.pipelines.base import Pipeline as _PipelineBase
        _transformers.Pipeline = _PipelineBase
    except Exception:
        pass
del _transformers

import torch
import whisperx

from utils import (
    get_compute_type,
    get_device,
    preprocess_audio,
    save_outputs,
    segments_to_text,
    fmt_ts,
)


# ────────────────────────────────────────────────────────────
# 1.  TRANSCRIPTION  (WhisperX)
# ────────────────────────────────────────────────────────────
def transcribe(
    audio_path: str,
    model_name: str = "large-v3",
    language: str | None = None,
    device: str = "cpu",
    compute_type: str = "float32",
    batch_size: int = 16,
) -> tuple[dict[str, Any], Any]:
    """
    Run WhisperX transcription + forced alignment.

    Returns
    -------
    result : dict with keys "segments" and "word_segments"
    model  : loaded WhisperX model (kept in memory for alignment)
    """
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Step 1 / 3  —  Transcription  (WhisperX)       ║")
    print("╚══════════════════════════════════════════════════╝")
    t0 = time.time()

    # Adjust batch size for CPU (avoid OOM)
    if device == "cpu":
        batch_size = min(batch_size, 4)

    print(f"  • Model        : {model_name}")
    print(f"  • Device       : {device}  ({compute_type})")
    print(f"  • Batch size   : {batch_size}")
    print(f"  • Language     : {language or 'auto-detect'}")

    # Load Whisper model
    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    # Load audio and transcribe
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)

    detected_lang = result.get("language", language or "??")
    print(f"  • Detected language : {detected_lang}")
    print(f"  • Raw segments      : {len(result['segments'])}")

    # ── Forced alignment (word-level timestamps) ───────────
    print("  • Running forced alignment …")
    align_model, align_meta = whisperx.load_align_model(
        language_code=detected_lang,
        device=device,
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        align_meta,
        audio,
        device=device,
        return_char_alignments=False,
    )

    elapsed = time.time() - t0
    print(f"  ✓ Transcription + alignment done in {elapsed:.1f}s")
    return result, audio


# ────────────────────────────────────────────────────────────
# 2.  SPEAKER DIARIZATION  (pyannote community model)
# ────────────────────────────────────────────────────────────
def diarize(
    audio_path: str,
    hf_token: str | None = None,
    device: str = "cpu",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
):
    """
    Run pyannote speaker-diarization-community-1.

    Returns
    -------
    diarize_segments : whisperx-compatible diarization result
    """
    from pyannote.audio import Pipeline as PyannotePipeline

    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Step 2 / 3  —  Speaker Diarization  (pyannote) ║")
    print("╚══════════════════════════════════════════════════╝")
    t0 = time.time()

    # Resolve HuggingFace token
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "  ⚠  No HuggingFace token found.\n"
            "     The community diarization model requires you to accept the\n"
            "     model conditions on HuggingFace and provide a token.\n"
            "     Set env var HF_TOKEN or pass --hf-token.\n"
            "     → https://huggingface.co/pyannote/speaker-diarization-community-1\n"
        )
        sys.exit(1)

    print("  • Loading pyannote/speaker-diarization-community-1 …")
    diarize_pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        use_auth_token=token,
    )

    # Move pipeline to device
    if device == "cuda":
        diarize_pipeline.to(torch.device("cuda"))
    elif device == "mps":
        try:
            diarize_pipeline.to(torch.device("mps"))
        except Exception:
            print("  ⚠  MPS not supported by pyannote, falling back to CPU")

    # Build kwargs for speaker hints
    diarize_kwargs: dict[str, Any] = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers

    if diarize_kwargs:
        print(f"  • Speaker hints : {diarize_kwargs}")

    print("  • Running diarization …")
    diarize_result = diarize_pipeline(audio_path, **diarize_kwargs)

    # Convert to whisperx-compatible format using whisperx's helper
    diarize_segments = whisperx.DiarizationPipeline.assign_word_speakers(
        diarize_result, {"segments": []}  # we only need the mapping
    )

    elapsed = time.time() - t0
    print(f"  ✓ Diarization done in {elapsed:.1f}s")
    return diarize_result


# ────────────────────────────────────────────────────────────
# 3.  MERGE  transcription + diarization
# ────────────────────────────────────────────────────────────
def merge(
    transcript_result: dict[str, Any],
    diarize_result,
) -> list[dict[str, Any]]:
    """
    Assign speaker labels to each transcription segment/word.

    Uses whisperx.assign_word_speakers under the hood.

    Returns
    -------
    List of segment dicts with keys: speaker, start, end, text
    """
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Step 3 / 3  —  Merging results                 ║")
    print("╚══════════════════════════════════════════════════╝")
    t0 = time.time()

    merged = whisperx.assign_word_speakers(diarize_result, transcript_result)

    # Build clean output segments
    output_segments: list[dict[str, Any]] = []
    for seg in merged.get("segments", []):
        output_segments.append({
            "speaker": seg.get("speaker", "UNKNOWN"),
            "start": round(seg.get("start", 0.0), 3),
            "end": round(seg.get("end", 0.0), 3),
            "text": seg.get("text", "").strip(),
            "words": [
                {
                    "word": w.get("word", ""),
                    "start": round(w.get("start", 0.0), 3),
                    "end": round(w.get("end", 0.0), 3),
                    "speaker": w.get("speaker", seg.get("speaker", "UNKNOWN")),
                }
                for w in seg.get("words", [])
                if "start" in w  # skip words without timestamps
            ],
        })

    # Rename speakers to friendly labels (SPEAKER_00 → Speaker 1)
    speaker_map: dict[str, str] = {}
    counter = 1
    for seg in output_segments:
        raw = seg["speaker"]
        if raw not in speaker_map:
            speaker_map[raw] = f"Speaker {counter}"
            counter += 1
        seg["speaker"] = speaker_map[raw]
        for w in seg.get("words", []):
            w["speaker"] = speaker_map.get(w["speaker"], w["speaker"])

    elapsed = time.time() - t0
    print(f"  • Detected {len(speaker_map)} speaker(s): {list(speaker_map.values())}")
    print(f"  • Output segments: {len(output_segments)}")
    print(f"  ✓ Merge done in {elapsed:.1f}s")

    return output_segments


# ────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Speech-to-Text + Speaker Diarization pipeline "
        "(WhisperX + pyannote community model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py meeting.wav
  python pipeline.py meeting.mp3 --language en --model large-v3
  python pipeline.py call.m4a --num-speakers 2 --hf-token hf_XXXXX
        """,
    )
    p.add_argument("audio", help="Path to audio/video file")
    p.add_argument(
        "--model", "-m",
        default="large-v3",
        help="Whisper model size (tiny, base, small, medium, large-v2, large-v3). "
             "Default: large-v3",
    )
    p.add_argument(
        "--language", "-l",
        default=None,
        help="Language code (en, id, ja, …). Default: auto-detect",
    )
    p.add_argument("--batch-size", "-b", type=int, default=16, help="Batch size for transcription")
    p.add_argument("--device", default=None, help="Force device: cuda, mps, cpu")
    p.add_argument("--hf-token", default=None, help="HuggingFace access token")
    p.add_argument("--num-speakers", type=int, default=None, help="Exact number of speakers (optional)")
    p.add_argument("--min-speakers", type=int, default=None, help="Minimum speakers (optional)")
    p.add_argument("--max-speakers", type=int, default=None, help="Maximum speakers (optional)")
    p.add_argument("--output-dir", "-o", default=None, help="Output directory (default: same as input)")
    p.add_argument("--no-preprocess", action="store_true", help="Skip ffmpeg preprocessing")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # ── Resolve device ──────────────────────────────────
    device = args.device or get_device()
    compute_type = get_compute_type(device)

    print("=" * 54)
    print("  Diarization Pipeline")
    print("  WhisperX + pyannote/speaker-diarization-community-1")
    print("=" * 54)
    print(f"  Input    : {args.audio}")
    print(f"  Device   : {device}")

    # ── Step 0: Preprocess audio ────────────────────────
    if args.no_preprocess:
        audio_path = str(Path(args.audio).resolve())
    else:
        audio_path = str(preprocess_audio(args.audio, args.output_dir))

    # ── Step 1: Transcribe ──────────────────────────────
    transcript_result, audio = transcribe(
        audio_path=audio_path,
        model_name=args.model,
        language=args.language,
        device=device,
        compute_type=compute_type,
        batch_size=args.batch_size,
    )

    # ── Step 2: Diarize ────────────────────────────────
    diarize_result = diarize(
        audio_path=audio_path,
        hf_token=args.hf_token,
        device=device,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    # ── Step 3: Merge ──────────────────────────────────
    segments = merge(transcript_result, diarize_result)

    # ── Step 4: Save outputs ───────────────────────────
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.audio).resolve().parent
    out_base = out_dir / f"{Path(args.audio).stem}_diarized"
    txt_path, json_path = save_outputs(segments, out_base)

    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Done!                                           ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  📝 Text output : {txt_path}")
    print(f"  📋 JSON output : {json_path}")

    # Print a preview of the transcript
    print("\n── Preview (first 15 segments) ──────────────────\n")
    text = segments_to_text(segments[:15])
    print(text)
    if len(segments) > 15:
        print(f"\n  … and {len(segments) - 15} more segment(s). See full output files.")
    print()


if __name__ == "__main__":
    main()
