"""
FastAPI backend for Speech Diarization pipeline.

Provides:
  - POST /api/upload         — upload + preprocess audio
  - POST /api/pipeline/run   — SSE endpoint streaming pipeline progress
  - GET  /api/config/devices — available compute devices
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
from pathlib import Path
from typing import Any
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", message=".*torchaudio.*backend.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Compatibility shim for transformers >= 4.42
import transformers as _transformers
if not hasattr(_transformers, "Pipeline"):
    try:
        from transformers.pipelines.base import Pipeline as _PipelineBase
        _transformers.Pipeline = _PipelineBase
    except Exception:
        pass
del _transformers

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="Diarization API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory stores & Constants
# ---------------------------------------------------------------------------
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "diarization_jobs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

_jobs: dict[str, dict] = {}      # job_id -> {audio_path, filename, ...}
_results: dict[str, dict] = {}   # job_id -> {segments, timings}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preprocess_audio(src_path: str) -> str:
    """Convert to 16 kHz mono WAV using ffmpeg."""
    wav_path = src_path.rsplit(".", 1)[0] + "_16k.wav"
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg not found on PATH")
    cmd = [
        ffmpeg_bin, "-y",
        "-i", src_path,
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        "-f", "wav",
        wav_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr}")
    return wav_path


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/config/devices")
async def get_devices():
    """Return list of available compute devices."""
    import torch
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.insert(0, "cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.insert(0, "mps")
    return {"devices": devices}


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and preprocess audio file."""
    job_id = str(uuid.uuid4())
    tmp_dir = tempfile.mkdtemp(prefix=f"diarize_{job_id[:8]}_")
    src_path = os.path.join(tmp_dir, file.filename)

    with open(src_path, "wb") as f:
        content = await file.read()
        f.write(content)

    file_size = len(content)

    try:
        wav_path = _preprocess_audio(src_path)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    _jobs[job_id] = {
        "audio_path": wav_path,
        "filename": file.filename,
        "file_size": file_size,
        "tmp_dir": tmp_dir,
    }

    return {"job_id": job_id, "filename": file.filename, "file_size": file_size}


@app.post("/api/pipeline/run")
async def run_pipeline_sse(
    job_id: str = Form(...),
    model_name: str = Form("large-v3"),
    language: str = Form("auto"),
    device: str = Form("cpu"),
    batch_size: int = Form(16),
    hf_token: str = Form(...),
    num_speakers: int = Form(None),
    min_speakers: int = Form(None),
    max_speakers: int = Form(None),
):
    """Run the full pipeline with SSE progress updates."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    audio_path = job["audio_path"]
    lang_val = None if language == "auto" else language
    compute_type = "float16" if device == "cuda" else "float32"

    async def event_generator():
        # Setup job configuration for the worker
        job_config = {
            "audio_path": audio_path,
            "hf_token": hf_token,
            "model_name": model_name,
            "device": device,
            "compute_type": compute_type,
            "batch_size": batch_size,
            "language": lang_val,
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers
        }
        
        job_file_path = os.path.join(UPLOAD_DIR, f"{job_id}.json")
        with open(job_file_path, "w", encoding="utf-8") as f:
            json.dump(job_config, f)

        worker_script = os.path.join(os.path.dirname(__file__), "worker.py")
        
        # Execute the worker in a separate process
        process = await asyncio.create_subprocess_exec(
            sys.executable, worker_script, job_file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )

        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                
                decoded_line = line.decode("utf-8").strip()
                if not decoded_line:
                    continue
                
                # Check if it's a valid JSON event
                if decoded_line.startswith('{"event":'):
                    try:
                        event_data = json.loads(decoded_line)
                        yield event_data
                    except json.JSONDecodeError:
                        print(f"[Worker Log] {decoded_line}")
                else:
                    # Print standard output/errors from the worker
                    print(f"[Worker Log] {decoded_line}")
            
            await process.wait()
            
            if process.returncode != 0:
                yield {"event": "error", "data": json.dumps({"detail": f"Worker process crashed with exit code {process.returncode}"})}
            
            # Load final result if it exists
            result_file_path = job_file_path + ".result.json"
            if os.path.exists(result_file_path):
                with open(result_file_path, "r", encoding="utf-8") as f:
                    final_result = json.load(f)
                    _results[job_id] = final_result
                    
        except asyncio.CancelledError:
            process.terminate()
            raise

    return EventSourceResponse(event_generator())


@app.get("/api/jobs/{job_id}/results")
async def get_results(job_id: str):
    """Retrieve cached pipeline results."""
    if job_id not in _results:
        raise HTTPException(status_code=404, detail="Results not found")
    return _results[job_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
