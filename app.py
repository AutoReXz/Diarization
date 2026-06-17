"""
app.py -- Streamlit UI for Speech-to-Text + Speaker Diarization Pipeline.

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", message=".*torchaudio.*backend.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Compatibility shim: transformers >= 4.42 no longer re-exports the Pipeline
# base class at the top level.  Both whisperx (asr.py) and some pyannote
# internals do `from transformers import Pipeline`, so we restore the export
# here before any of those modules are first imported.
# ---------------------------------------------------------------------------
import transformers as _transformers
if not hasattr(_transformers, "Pipeline"):
    try:
        from transformers.pipelines.base import Pipeline as _PipelineBase
        _transformers.Pipeline = _PipelineBase
    except Exception:
        pass
del _transformers

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Speech Diarization",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS -- Professional dark-mode inspired design, no emoji
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- Google Fonts ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ---------- Root Tokens ---------- */
    :root {
        --bg-primary: #0f1117;
        --bg-secondary: #161b22;
        --bg-tertiary: #1c2333;
        --bg-card: rgba(22, 27, 34, 0.85);
        --bg-card-hover: rgba(28, 35, 51, 0.95);
        --bg-glass: rgba(255, 255, 255, 0.03);
        --border-primary: rgba(255, 255, 255, 0.06);
        --border-subtle: rgba(255, 255, 255, 0.04);
        --border-accent: rgba(99, 135, 241, 0.3);
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --text-muted: #545d68;
        --accent-blue: #6387f1;
        --accent-blue-soft: rgba(99, 135, 241, 0.12);
        --accent-green: #3fb950;
        --accent-green-soft: rgba(63, 185, 80, 0.12);
        --accent-red: #f85149;
        --accent-red-soft: rgba(248, 81, 73, 0.12);
        --accent-amber: #d29922;
        --accent-amber-soft: rgba(210, 153, 34, 0.12);
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
        --radius-xl: 20px;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.35), 0 2px 4px rgba(0,0,0,0.25);
        --shadow-lg: 0 10px 30px rgba(0,0,0,0.4), 0 4px 10px rgba(0,0,0,0.3);
        --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ---------- Global Overrides ---------- */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* ---------- Header ---------- */
    .app-header {
        padding: 1.5rem 0 1.25rem 0;
        margin-bottom: 1.75rem;
        border-bottom: 1px solid var(--border-primary);
        position: relative;
    }
    .app-header::after {
        content: '';
        position: absolute;
        bottom: -1px;
        left: 0;
        width: 120px;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-blue), transparent);
        border-radius: 1px;
    }
    .app-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin: 0;
        color: var(--text-primary);
        line-height: 1.3;
    }
    .app-header .subtitle {
        font-size: 0.88rem;
        font-weight: 400;
        color: var(--text-secondary);
        margin: 0.35rem 0 0 0;
        line-height: 1.5;
    }
    .app-header .tech-stack {
        display: inline-flex;
        gap: 6px;
        margin-top: 0.75rem;
    }
    .app-header .tech-tag {
        font-size: 0.68rem;
        font-weight: 500;
        letter-spacing: 0.03em;
        padding: 3px 10px;
        border-radius: 100px;
        background: var(--bg-glass);
        border: 1px solid var(--border-primary);
        color: var(--text-muted);
        text-transform: uppercase;
    }

    /* ---------- Status Cards ---------- */
    .status-card {
        background: var(--bg-card);
        border: 1px solid var(--border-primary);
        border-radius: var(--radius-md);
        padding: 0.9rem 1.15rem;
        margin-bottom: 0.6rem;
        backdrop-filter: blur(12px);
        transition: var(--transition);
    }
    .status-card.active {
        border-left: 3px solid var(--accent-blue);
        background: var(--accent-blue-soft);
    }
    .status-card.done {
        border-left: 3px solid var(--accent-green);
        background: var(--accent-green-soft);
    }
    .status-card.error {
        border-left: 3px solid var(--accent-red);
        background: var(--accent-red-soft);
    }
    .status-card .step-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.2rem;
    }
    .status-card .step-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    .status-card .step-detail {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }
    .status-card.active .step-title::before {
        content: '';
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--accent-blue);
        margin-right: 8px;
        vertical-align: middle;
        animation: pulse-dot 1.5s infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* ---------- Metric Cards ---------- */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-primary);
        border-radius: var(--radius-md);
        padding: 1.15rem 1rem;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        border-color: var(--border-accent);
        transform: translateY(-1px);
        box-shadow: var(--shadow-sm);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .metric-card:hover::before {
        opacity: 1;
    }
    .metric-card .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.55rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .metric-card .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.35rem;
    }

    /* ---------- Transcript Rows ---------- */
    .transcript-container {
        border: 1px solid var(--border-primary);
        border-radius: var(--radius-lg);
        overflow: hidden;
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        max-height: 640px;
        overflow-y: auto;
    }
    .transcript-container::-webkit-scrollbar {
        width: 6px;
    }
    .transcript-container::-webkit-scrollbar-track {
        background: transparent;
    }
    .transcript-container::-webkit-scrollbar-thumb {
        background: var(--border-primary);
        border-radius: 3px;
    }
    .transcript-container::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.12);
    }
    .transcript-row {
        display: flex;
        gap: 0.85rem;
        padding: 0.85rem 1.15rem;
        border-bottom: 1px solid var(--border-subtle);
        align-items: flex-start;
        transition: background 0.15s ease;
    }
    .transcript-row:last-child {
        border-bottom: none;
    }
    .transcript-row:hover {
        background: rgba(255, 255, 255, 0.02);
    }
    .transcript-row .ts {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-muted);
        white-space: nowrap;
        min-width: 140px;
        padding-top: 3px;
        letter-spacing: -0.01em;
    }
    .transcript-row .speaker-badge {
        display: inline-block;
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 12px;
        border-radius: 100px;
        white-space: nowrap;
        min-width: 88px;
        text-align: center;
        letter-spacing: 0.01em;
    }
    .transcript-row .text {
        font-family: 'Inter', sans-serif;
        font-size: 0.88rem;
        color: var(--text-primary);
        line-height: 1.6;
        flex: 1;
        font-weight: 400;
    }

    /* Speaker badge colour palette -- muted, professional tones */
    .spk-1  { background: rgba(96, 165, 250, 0.15); color: #93bbfd; border: 1px solid rgba(96,165,250,0.2); }
    .spk-2  { background: rgba(244, 114, 182, 0.12); color: #f0a0c8; border: 1px solid rgba(244,114,182,0.18); }
    .spk-3  { background: rgba(52, 211, 153, 0.12); color: #6ee7b7; border: 1px solid rgba(52,211,153,0.18); }
    .spk-4  { background: rgba(251, 191, 36, 0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.18); }
    .spk-5  { background: rgba(167, 139, 250, 0.12); color: #c4b5fd; border: 1px solid rgba(167,139,250,0.18); }
    .spk-6  { background: rgba(251, 146, 60, 0.12); color: #fdba74; border: 1px solid rgba(251,146,60,0.18); }
    .spk-7  { background: rgba(129, 140, 248, 0.12); color: #a5b4fc; border: 1px solid rgba(129,140,248,0.18); }
    .spk-8  { background: rgba(192, 132, 252, 0.12); color: #d8b4fe; border: 1px solid rgba(192,132,252,0.18); }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--border-primary);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-top: 1.25rem;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    .sidebar-footer {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        color: var(--text-muted);
        text-align: center;
        padding-top: 0.5rem;
        border-top: 1px solid var(--border-subtle);
        line-height: 1.6;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid var(--border-primary);
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.7rem 1.35rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 0.01em;
        color: var(--text-secondary);
        border-bottom: 2px solid transparent;
        transition: var(--transition);
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-blue) !important;
        border-bottom-color: var(--accent-blue) !important;
    }

    /* ---------- Speaker Stats Card ---------- */
    .speaker-stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border-primary);
        border-radius: var(--radius-md);
        padding: 1.1rem;
        transition: var(--transition);
        backdrop-filter: blur(12px);
    }
    .speaker-stat-card:hover {
        border-color: var(--border-accent);
    }
    .speaker-stat-card .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.25rem 0;
        font-size: 0.82rem;
    }
    .speaker-stat-card .stat-row .stat-key {
        color: var(--text-secondary);
        font-weight: 400;
    }
    .speaker-stat-card .stat-row .stat-val {
        color: var(--text-primary);
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }

    /* ---------- Timeline ---------- */
    .timeline-bar {
        position: relative;
        height: 28px;
        background: var(--bg-tertiary);
        border-radius: var(--radius-sm);
        overflow: hidden;
        border: 1px solid var(--border-primary);
    }
    .timeline-seg {
        position: absolute;
        height: 100%;
        border-radius: 2px;
        transition: opacity 0.2s;
        cursor: default;
    }
    .timeline-seg:hover {
        opacity: 0.85 !important;
    }
    .timeline-labels {
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: var(--text-muted);
        margin-top: 5px;
        padding: 0 2px;
    }
    .timeline-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        margin-top: 10px;
    }
    .timeline-legend-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.76rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    .timeline-legend-swatch {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 3px;
    }

    /* ---------- Empty State ---------- */
    .empty-state {
        text-align: center;
        padding: 5rem 2rem;
    }
    .empty-state .empty-icon {
        width: 64px;
        height: 64px;
        margin: 0 auto 1.25rem auto;
        border-radius: var(--radius-lg);
        background: var(--bg-glass);
        border: 1px solid var(--border-primary);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .empty-state .empty-icon svg {
        width: 28px;
        height: 28px;
        color: var(--text-muted);
        opacity: 0.5;
    }
    .empty-state .empty-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 0.4rem;
    }
    .empty-state .empty-desc {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        color: var(--text-muted);
        line-height: 1.6;
    }

    /* ---------- File Info ---------- */
    .file-info {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        color: var(--text-muted);
        padding: 0.5rem 0;
    }
    .file-info span {
        color: var(--text-secondary);
    }

    /* ---------- Section Heading ---------- */
    .section-heading {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid var(--border-subtle);
    }

    /* ---------- Filter Bar ---------- */
    .filter-count {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }

    /* ---------- Divider ---------- */
    .section-divider {
        border: none;
        border-top: 1px solid var(--border-primary);
        margin: 1.5rem 0;
    }

    /* ---------- Hide Streamlit Branding ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_ts(seconds: float) -> str:
    """Format seconds to HH:MM:SS.mmm"""
    td = timedelta(seconds=seconds)
    total_secs = int(td.total_seconds())
    hrs, remainder = divmod(total_secs, 3600)
    mins, secs = divmod(remainder, 60)
    millis = int((seconds - total_secs) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def fmt_duration(seconds: float) -> str:
    """Human-readable duration."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def speaker_badge_class(speaker: str) -> str:
    """Return CSS class for a speaker label."""
    try:
        num = int(speaker.split()[-1])
    except (ValueError, IndexError):
        num = 1
    idx = ((num - 1) % 8) + 1
    return f"spk-{idx}"


def segments_to_text(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        start = fmt_ts(seg["start"])
        end = fmt_ts(seg["end"])
        text = seg["text"].strip()
        lines.append(f"[{start} -> {end}]  {speaker}:  {text}")
    return "\n".join(lines)


def segments_to_json(segments: list[dict], pretty: bool = True) -> str:
    indent = 2 if pretty else None
    return json.dumps(segments, indent=indent, ensure_ascii=False)


def status_card(label: str, title: str, detail: str = "", state: str = "") -> str:
    """Render a status card (state: '', 'active', 'done', 'error')."""
    cls = f"status-card {state}".strip()
    detail_html = f'<div class="step-detail">{detail}</div>' if detail else ""
    return f"""
    <div class="{cls}">
        <div class="step-label">{label}</div>
        <div class="step-title">{title}</div>
        {detail_html}
    </div>
    """


# ---------------------------------------------------------------------------
# Pipeline runner  (wraps pipeline.py functions with Streamlit progress)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_name: str, device: str, compute_type: str, language: str | None):
    """Cache the Whisper model across reruns."""
    import whisperx
    return whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=language,
    )


def get_diarization_pipeline(hf_token: str, device: str):
    """
    Load (or reuse from session_state) the pyannote diarization pipeline.

    Intentionally NOT wrapped in @st.cache_resource because Streamlit's
    cache serialisation conflicts with pyannote's dynamic class loader,
    causing "Could not import module 'Pipeline'" errors.
    """
    import torch
    # Build a stable cache key so we reload only when settings change
    cache_key = f"diarize_pipe_{device}"
    cached = st.session_state.get(cache_key)
    if cached is not None:
        return cached

    # Import here to keep the call-stack clean for pyannote's class resolver
    from pyannote.audio import Pipeline as PyannotePipeline

    pipe = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )
    if device == "cuda":
        pipe.to(torch.device("cuda"))
    elif device == "mps":
        try:
            pipe.to(torch.device("mps"))
        except Exception:
            pass  # pyannote may not support MPS; falls back to CPU silently

    st.session_state[cache_key] = pipe
    return pipe


def run_pipeline(
    audio_path: str,
    model_name: str,
    language: str | None,
    device: str,
    compute_type: str,
    batch_size: int,
    hf_token: str,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    progress_container,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """
    Execute the full pipeline, updating the Streamlit progress_container
    at each step.

    Returns (segments, timing_dict).
    """
    import whisperx

    timings: dict[str, float] = {}

    # -- Step 1: Load model & transcribe --
    with progress_container:
        step1_ph = st.empty()
        step1_ph.markdown(
            status_card("Step 1 of 4", "Loading ASR model and transcribing", state="active"),
            unsafe_allow_html=True,
        )

    t0 = time.time()
    if device == "cpu":
        batch_size = min(batch_size, 4)

    model = load_whisper_model(model_name, device, compute_type, language)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)
    detected_lang = result.get("language", language or "unknown")
    timings["transcription"] = time.time() - t0

    with progress_container:
        step1_ph.markdown(
            status_card(
                "Step 1 of 4",
                "Transcription complete",
                f"Language: {detected_lang}  /  Segments: {len(result['segments'])}  /  {timings['transcription']:.1f}s",
                state="done",
            ),
            unsafe_allow_html=True,
        )

    # -- Step 2: Forced alignment --
    with progress_container:
        step2_ph = st.empty()
        step2_ph.markdown(
            status_card("Step 2 of 4", "Running forced alignment", state="active"),
            unsafe_allow_html=True,
        )

    t0 = time.time()
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
    timings["alignment"] = time.time() - t0

    with progress_container:
        step2_ph.markdown(
            status_card(
                "Step 2 of 4",
                "Alignment complete",
                f"Word-level timestamps generated  /  {timings['alignment']:.1f}s",
                state="done",
            ),
            unsafe_allow_html=True,
        )

    # -- Step 3: Speaker diarization --
    with progress_container:
        step3_ph = st.empty()
        step3_ph.markdown(
            status_card("Step 3 of 4", "Running speaker diarization", state="active"),
            unsafe_allow_html=True,
        )

    t0 = time.time()
    diarize_pipe = get_diarization_pipeline(hf_token, device)
    diarize_kwargs: dict[str, Any] = {}
    if num_speakers is not None and num_speakers > 0:
        diarize_kwargs["num_speakers"] = num_speakers
    if min_speakers is not None and min_speakers > 0:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None and max_speakers > 0:
        diarize_kwargs["max_speakers"] = max_speakers

    import torch
    import pandas as pd
    # whisperx already loaded the audio as a 16kHz mono numpy array
    waveform = torch.from_numpy(audio).unsqueeze(0)
    audio_input = {"waveform": waveform, "sample_rate": 16000}
    diarize_result = diarize_pipe(audio_input, **diarize_kwargs)
    
    # Convert pyannote Annotation or DiarizeOutput to DataFrame for whisperX
    if hasattr(diarize_result, "speaker_diarization"):
        annotation = diarize_result.speaker_diarization
    else:
        annotation = diarize_result
        
    diarize_list = []
    if hasattr(annotation, "itertracks"):
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            diarize_list.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    elif hasattr(annotation, "segments"):
        for seg in annotation.segments:
            diarize_list.append({"start": seg.start, "end": seg.end, "speaker": seg.speaker})
    else:
        # Fallback if it's iterable
        for turn, _, speaker in annotation:
            diarize_list.append({"start": turn.start, "end": turn.end, "speaker": speaker})
            
    diarize_df = pd.DataFrame(diarize_list)

    timings["diarization"] = time.time() - t0

    with progress_container:
        step3_ph.markdown(
            status_card(
                "Step 3 of 4",
                "Diarization complete",
                f"{timings['diarization']:.1f}s",
                state="done",
            ),
            unsafe_allow_html=True,
        )

    # -- Step 4: Merge --
    with progress_container:
        step4_ph = st.empty()
        step4_ph.markdown(
            status_card("Step 4 of 4", "Merging transcription with speaker labels", state="active"),
            unsafe_allow_html=True,
        )

    t0 = time.time()
    merged = whisperx.assign_word_speakers(diarize_df, result)

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
                if "start" in w
            ],
        })

    # Rename speakers to friendly labels
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

    timings["merge"] = time.time() - t0
    timings["total"] = sum(timings.values())

    with progress_container:
        step4_ph.markdown(
            status_card(
                "Step 4 of 4",
                "Merge complete",
                f"Speakers: {len(speaker_map)}  /  Segments: {len(output_segments)}  /  {timings['merge']:.1f}s",
                state="done",
            ),
            unsafe_allow_html=True,
        )

    return output_segments, timings


# ---------------------------------------------------------------------------
# Audio preprocessing via ffmpeg
# ---------------------------------------------------------------------------
def preprocess_uploaded_audio(uploaded_file) -> str:
    """
    Save uploaded file to temp dir and convert to 16 kHz mono WAV.
    Returns path to the preprocessed WAV.
    """
    import shutil
    import subprocess

    tmp_dir = tempfile.mkdtemp(prefix="diarize_")
    original_path = os.path.join(tmp_dir, uploaded_file.name)

    with open(original_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    wav_path = os.path.join(tmp_dir, "audio_16k.wav")
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        st.error(
            "ffmpeg is not installed or not found on PATH. "
            "Please install ffmpeg before running the pipeline."
        )
        st.stop()

    cmd = [
        ffmpeg_bin, "-y",
        "-i", original_path,
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        "-f", "wav",
        wav_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        st.error(f"ffmpeg preprocessing failed:\n{proc.stderr}")
        st.stop()

    return wav_path


# ---------------------------------------------------------------------------
# Layout: Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Configuration")

    hf_token = st.text_input(
        "HuggingFace Token",
        value=os.environ.get("HF_TOKEN", ""),
        type="password",
        help="Required for pyannote diarization model. Get one at huggingface.co/settings/tokens",
    )

    st.markdown("---")
    st.markdown("### Model")

    model_name = st.selectbox(
        "Whisper Model",
        options=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        index=5,
        help="Larger models are more accurate but slower and require more memory.",
    )

    language = st.selectbox(
        "Language",
        options=[
            "auto",
            "en", "id", "zh", "ja", "ko", "de", "fr", "es",
            "pt", "ru", "ar", "hi", "it", "nl", "pl", "tr",
            "vi", "th", "sv", "da", "fi", "no", "uk", "cs",
        ],
        index=0,
        help="Select 'auto' for automatic language detection.",
    )
    language_val = None if language == "auto" else language

    st.markdown("---")
    st.markdown("### Speaker Settings")

    speaker_mode = st.radio(
        "Speaker count",
        options=["Auto-detect", "Exact count", "Range"],
        index=0,
        horizontal=True,
    )

    num_speakers = None
    min_speakers = None
    max_speakers = None

    if speaker_mode == "Exact count":
        num_speakers = st.number_input("Number of speakers", min_value=1, max_value=20, value=2)
    elif speaker_mode == "Range":
        col_min, col_max = st.columns(2)
        with col_min:
            min_speakers = st.number_input("Min speakers", min_value=1, max_value=20, value=2)
        with col_max:
            max_speakers = st.number_input("Max speakers", min_value=1, max_value=20, value=5)

    st.markdown("---")
    st.markdown("### Advanced")

    # Device detection
    import torch
    available_devices = ["cpu"]
    if torch.cuda.is_available():
        available_devices.insert(0, "cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        available_devices.insert(0, "mps")

    device = st.selectbox(
        "Compute Device",
        options=available_devices,
        index=0,
        help="GPU (cuda/mps) is strongly recommended for large models.",
    )

    batch_size = st.slider(
        "Batch Size",
        min_value=1,
        max_value=32,
        value=16 if device != "cpu" else 4,
        help="Higher values use more memory but process faster.",
    )

    compute_type = "float16" if device == "cuda" else "float32"

    st.markdown("---")
    st.markdown(
        '<div class="sidebar-footer">'
        "WhisperX + pyannote community model<br>Self-hosted / open-source"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Layout: Main content area
# ---------------------------------------------------------------------------

# Header
st.markdown(
    """
    <div class="app-header">
        <h1>Speech-to-Text with Speaker Diarization</h1>
        <p class="subtitle">
            Upload an audio file to transcribe with automatic speaker identification.
            Fully self-hosted, powered by WhisperX and pyannote.
        </p>
        <div class="tech-stack">
            <span class="tech-tag">WhisperX</span>
            <span class="tech-tag">pyannote</span>
            <span class="tech-tag">Open Source</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# File uploader
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "m4a", "flac", "ogg", "wma", "aac", "mp4", "webm", "mkv"],
    help="Supported formats: WAV, MP3, M4A, FLAC, OGG, WMA, AAC, MP4, WebM, MKV",
)

if uploaded_file is not None:
    # Show audio player
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.markdown(
        f'<div class="file-info">File: <span>{uploaded_file.name}</span>'
        f' &nbsp;/&nbsp; Size: <span>{file_size_mb:.1f} MB</span></div>',
        unsafe_allow_html=True,
    )

    # Validate token
    if not hf_token:
        st.warning(
            "A HuggingFace token is required for speaker diarization. "
            "Enter your token in the sidebar to proceed."
        )

    # Run button
    run_disabled = not hf_token
    if st.button("Run Pipeline", type="primary", disabled=run_disabled, use_container_width=True):

        # -- Preprocess --
        with st.spinner("Preprocessing audio..."):
            wav_path = preprocess_uploaded_audio(uploaded_file)

        # -- Progress container --
        progress_container = st.container()

        try:
            segments, timings = run_pipeline(
                audio_path=wav_path,
                model_name=model_name,
                language=language_val,
                device=device,
                compute_type=compute_type,
                batch_size=batch_size,
                hf_token=hf_token,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                progress_container=progress_container,
            )

            # Store results in session state
            st.session_state["segments"] = segments
            st.session_state["timings"] = timings
            st.session_state["source_file"] = uploaded_file.name
            st.session_state["pipeline_done"] = True

        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
            st.exception(e)


# ---------------------------------------------------------------------------
# Results display (persists across reruns via session_state)
# ---------------------------------------------------------------------------
if st.session_state.get("pipeline_done"):
    segments = st.session_state["segments"]
    timings = st.session_state["timings"]
    source_file = st.session_state["source_file"]

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # -- Summary metrics --
    speakers = sorted(set(seg["speaker"] for seg in segments))
    total_duration = max((seg["end"] for seg in segments), default=0)
    total_words = sum(len(seg.get("words", [])) for seg in segments)

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, str(len(segments)), "Segments"),
        (c2, str(len(speakers)), "Speakers"),
        (c3, str(total_words), "Words"),
        (c4, fmt_duration(total_duration), "Duration"),
        (c5, f"{timings['total']:.1f}s", "Processing"),
    ]
    for col, value, label in metrics:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # -- Tabs: Transcript / JSON / Statistics --
    tab_transcript, tab_json, tab_stats = st.tabs(["Transcript", "JSON", "Statistics"])

    # -- TAB: Transcript --
    with tab_transcript:
        # Speaker filter
        filter_col1, filter_col2 = st.columns([3, 1])
        with filter_col1:
            selected_speakers = st.multiselect(
                "Filter by speaker",
                options=speakers,
                default=speakers,
            )
        with filter_col2:
            search_query = st.text_input("Search text", placeholder="Search...")

        filtered = [
            seg for seg in segments
            if seg["speaker"] in selected_speakers
            and (not search_query or search_query.lower() in seg["text"].lower())
        ]

        st.markdown(
            f'<div class="filter-count">'
            f"Showing {len(filtered)} of {len(segments)} segments</div>",
            unsafe_allow_html=True,
        )

        # Render transcript
        transcript_html_parts = []
        for seg in filtered:
            ts_start = fmt_ts(seg["start"])
            ts_end = fmt_ts(seg["end"])
            badge_cls = speaker_badge_class(seg["speaker"])
            transcript_html_parts.append(
                f'<div class="transcript-row">'
                f'  <div class="ts">{ts_start} &rarr; {ts_end}</div>'
                f'  <div><span class="speaker-badge {badge_cls}">{seg["speaker"]}</span></div>'
                f'  <div class="text">{seg["text"]}</div>'
                f"</div>"
            )

        if transcript_html_parts:
            inner = "\n".join(transcript_html_parts)
            st.markdown(
                f'<div class="transcript-container">{inner}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No segments match the current filter.")

        # Download buttons
        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
        dl_col1, dl_col2, _ = st.columns([1, 1, 3])
        base_name = Path(source_file).stem
        with dl_col1:
            st.download_button(
                label="Download TXT",
                data=segments_to_text(segments),
                file_name=f"{base_name}_diarized.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                label="Download JSON",
                data=segments_to_json(segments),
                file_name=f"{base_name}_diarized.json",
                mime="application/json",
                use_container_width=True,
            )

    # -- TAB: JSON --
    with tab_json:
        st.code(segments_to_json(segments), language="json")

    # -- TAB: Statistics --
    with tab_stats:
        st.markdown('<div class="section-heading">Processing Time Breakdown</div>', unsafe_allow_html=True)
        timing_cols = st.columns(4)
        timing_labels = [
            ("Transcription", "transcription"),
            ("Alignment", "alignment"),
            ("Diarization", "diarization"),
            ("Merge", "merge"),
        ]
        for col, (label, key) in zip(timing_cols, timing_labels):
            with col:
                val = timings.get(key, 0)
                pct = (val / timings["total"] * 100) if timings["total"] > 0 else 0
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{val:.1f}s</div>'
                    f'<div class="metric-label">{label} ({pct:.0f}%)</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)

        # Per-speaker breakdown
        st.markdown('<div class="section-heading">Speaker Breakdown</div>', unsafe_allow_html=True)

        speaker_data: dict[str, dict] = {}
        for seg in segments:
            spk = seg["speaker"]
            if spk not in speaker_data:
                speaker_data[spk] = {"segments": 0, "words": 0, "duration": 0.0}
            speaker_data[spk]["segments"] += 1
            speaker_data[spk]["words"] += len(seg.get("words", []))
            speaker_data[spk]["duration"] += seg["end"] - seg["start"]

        spk_cols = st.columns(min(len(speaker_data), 4))
        for i, (spk, data) in enumerate(sorted(speaker_data.items())):
            col = spk_cols[i % len(spk_cols)]
            badge_cls = speaker_badge_class(spk)
            dur_pct = (data["duration"] / total_duration * 100) if total_duration > 0 else 0
            with col:
                st.markdown(
                    f'<div class="speaker-stat-card">'
                    f'<div style="margin-bottom:0.65rem;">'
                    f'<span class="speaker-badge {badge_cls}" '
                    f'style="font-size:0.78rem; padding:4px 14px;">{spk}</span></div>'
                    f'<div class="stat-row"><span class="stat-key">Segments</span>'
                    f'<span class="stat-val">{data["segments"]}</span></div>'
                    f'<div class="stat-row"><span class="stat-key">Words</span>'
                    f'<span class="stat-val">{data["words"]}</span></div>'
                    f'<div class="stat-row"><span class="stat-key">Duration</span>'
                    f'<span class="stat-val">{fmt_duration(data["duration"])}</span></div>'
                    f'<div class="stat-row"><span class="stat-key">Share</span>'
                    f'<span class="stat-val">{dur_pct:.1f}%</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)

        # Timeline view
        st.markdown('<div class="section-heading">Speaker Timeline</div>', unsafe_allow_html=True)
        if total_duration > 0:
            color_map = {
                "spk-1": "#60a5fa", "spk-2": "#f472b6", "spk-3": "#34d399",
                "spk-4": "#fbbf24", "spk-5": "#a78bfa", "spk-6": "#fb923c",
                "spk-7": "#818cf8", "spk-8": "#c084fc",
            }

            timeline_parts = []
            for seg in segments:
                width_pct = (seg["end"] - seg["start"]) / total_duration * 100
                left_pct = seg["start"] / total_duration * 100
                badge_cls = speaker_badge_class(seg["speaker"])
                color = color_map.get(badge_cls, "#60a5fa")
                timeline_parts.append(
                    f'<div class="timeline-seg" '
                    f'title="{seg["speaker"]}: {fmt_ts(seg["start"])} - {fmt_ts(seg["end"])}" '
                    f'style="left:{left_pct:.2f}%; width:{max(width_pct, 0.3):.2f}%; '
                    f'background:{color}; opacity:0.7;"></div>'
                )

            timeline_inner = "\n".join(timeline_parts)
            st.markdown(
                f'<div class="timeline-bar">{timeline_inner}</div>'
                f'<div class="timeline-labels">'
                f"<span>00:00:00</span><span>{fmt_ts(total_duration)}</span></div>",
                unsafe_allow_html=True,
            )

            # Legend
            legend_parts = []
            for spk in sorted(speaker_data.keys()):
                badge_cls = speaker_badge_class(spk)
                color = color_map.get(badge_cls, "#60a5fa")
                legend_parts.append(
                    f'<span class="timeline-legend-item">'
                    f'<span class="timeline-legend-swatch" style="background:{color};"></span>'
                    f'{spk}</span>'
                )
            st.markdown(
                f'<div class="timeline-legend">{"".join(legend_parts)}</div>',
                unsafe_allow_html=True,
            )

else:
    # Empty state
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
                     stroke-width="1.5" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round"
                          d="M12 18.75a6 6 0 0 0 6-6v-1.5m-6 7.5a6 6 0 0 1-6-6v-1.5m6 7.5v3.75m-3.75
                          0h7.5M12 15.75a3 3 0 0 1-3-3V4.5a3 3 0 1 1 6 0v8.25a3 3 0 0 1-3 3Z"/>
                </svg>
            </div>
            <div class="empty-title">Upload an audio file to get started</div>
            <div class="empty-desc">
                Supported formats: WAV, MP3, M4A, FLAC, OGG, AAC, MP4, WebM, MKV
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
