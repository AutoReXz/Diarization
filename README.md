# 🎙️ Speech-to-Text + Speaker Diarization Pipeline

**Fully self-hosted, 100% open-source** pipeline for transcribing audio with speaker labels.

| Component | Model | Source |
|---|---|---|
| ASR (Speech-to-Text) | [WhisperX](https://github.com/m-bain/whisperX) (OpenAI Whisper + forced alignment) | Open-source |
| Speaker Diarization | [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | Open-source (HuggingFace) |
| Audio Processing | ffmpeg | Open-source |
| ML Backend | PyTorch | Open-source |

**No paid APIs required.** Everything runs locally on your machine.

---

## 📁 Project Structure

```
Diarization/
├── pipeline.py           # Main pipeline: transcribe → diarize → merge → output
├── utils.py              # Helpers: audio preprocessing, device detection, formatting
├── requirements.txt      # Python dependencies
├── setup.sh              # One-shot setup script (Linux/macOS)
├── README.md             # This file
└── examples/
    ├── sample_output.json   # Example JSON output with word-level timestamps
    └── sample_output.txt    # Example plain-text transcript
```

---

## ⚙️ Installation

### Prerequisites

| Requirement | Version | Check |
|---|---|---|
| Python | 3.9 – 3.11 | `python3 --version` |
| ffmpeg | any recent | `ffmpeg -version` |
| Git | any | `git --version` |
| HuggingFace account | — | [huggingface.co](https://huggingface.co) |

### Step 1 — HuggingFace Token

The pyannote community model requires accepting model conditions:

1. Create a free account at [huggingface.co](https://huggingface.co/join)
2. Accept the conditions for these models:
   - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   - [pyannote/segmentation](https://huggingface.co/pyannote/segmentation)
3. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Step 2 — Quick Setup (Linux/macOS)

```bash
chmod +x setup.sh
./setup.sh
```

### Step 2 — Manual Setup (any OS)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel

# (Optional) Install PyTorch with CUDA for GPU acceleration
# Visit https://pytorch.org/get-started/locally/ for your specific CUDA version
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install -r requirements.txt
```

### Installing ffmpeg

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
winget install ffmpeg
# or: choco install ffmpeg
```

---

## 🚀 Usage

### Basic

```bash
# Set your HuggingFace token (one time)
export HF_TOKEN="hf_your_token_here"

# Run the pipeline
python pipeline.py meeting.wav
```

### Options

```bash
python pipeline.py audio.wav \
    --model large-v3 \          # Whisper model (tiny/base/small/medium/large-v2/large-v3)
    --language en \              # Language code (or omit for auto-detect)
    --num-speakers 3 \           # Exact number of speakers (optional)
    --min-speakers 2 \           # Minimum speakers (optional)
    --max-speakers 5 \           # Maximum speakers (optional)
    --batch-size 16 \            # Transcription batch size
    --device cuda \              # Force device (cuda/mps/cpu)
    --hf-token hf_XXXXX \       # HuggingFace token (or set HF_TOKEN env var)
    --output-dir ./results \     # Output directory
    --no-preprocess              # Skip ffmpeg conversion
```

### Examples

```bash
# English meeting, 2 speakers, fast model
python pipeline.py meeting.mp3 --model base --language en --num-speakers 2

# Indonesian call recording, auto-detect everything
python pipeline.py panggilan.m4a --language id

# Japanese interview, GPU, large model
python pipeline.py interview.wav --model large-v3 --language ja --device cuda

# Process a video file (ffmpeg will extract audio)
python pipeline.py webinar.mp4 --model medium
```

---

## 📄 Output Format

The pipeline produces two output files:

### Plain Text (`_diarized.txt`)

```
[00:00:00.521 → 00:00:04.873]  Speaker 1:  Selamat pagi semuanya, terima kasih sudah bergabung di meeting hari ini.
[00:00:05.241 → 00:00:08.193]  Speaker 2:  Pagi, siap. Saya sudah siapkan materinya.
[00:00:08.892 → 00:00:14.521]  Speaker 1:  Bagus. Jadi agenda hari ini kita akan bahas progress sprint kemarin dan planning untuk sprint berikutnya.
[00:00:15.102 → 00:00:19.413]  Speaker 3:  Saya mau update, fitur authentication sudah selesai dan sudah di-merge ke branch main.
[00:00:20.001 → 00:00:23.782]  Speaker 2:  Nice, dari sisi frontend saya masih ada satu bug yang perlu di-fix di halaman dashboard.
```

### JSON (`_diarized.json`)

```json
[
  {
    "speaker": "Speaker 1",
    "start": 0.521,
    "end": 4.873,
    "text": "Selamat pagi semuanya, terima kasih sudah bergabung di meeting hari ini.",
    "words": [
      { "word": "Selamat", "start": 0.521, "end": 0.891, "speaker": "Speaker 1" },
      { "word": "pagi",    "start": 0.911, "end": 1.201, "speaker": "Speaker 1" },
      ...
    ]
  },
  {
    "speaker": "Speaker 2",
    "start": 5.241,
    "end": 8.193,
    "text": "Pagi, siap. Saya sudah siapkan materinya.",
    "words": [ ... ]
  }
]
```

Each segment contains:
- **`speaker`** — Speaker label (Speaker 1, Speaker 2, …)
- **`start` / `end`** — Timestamp in seconds
- **`text`** — Full sentence transcription
- **`words`** — Word-level detail with individual timestamps and speaker assignment

---

## 🔧 Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT AUDIO                               │
│                  (any format: wav, mp3, m4a, mp4, …)            │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 0: PREPROCESSING  (ffmpeg)                                 │
│  Convert to 16 kHz, mono, 16-bit WAV                            │
└──────────────────────┬───────────────────────────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
┌─────────────────────┐ ┌─────────────────────────────────────────┐
│  Step 1: ASR        │ │  Step 2: DIARIZATION                    │
│  WhisperX           │ │  pyannote/speaker-diarization-          │
│  • Transcribe       │ │  community-1                            │
│  • Forced alignment │ │  • VAD → Segmentation → Clustering     │
│  • Word timestamps  │ │  • Speaker turn detection               │
└────────┬────────────┘ └──────────────┬──────────────────────────┘
         │                             │
         └──────────┬──────────────────┘
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 3: MERGE                                                   │
│  Assign speaker labels to each transcription segment/word        │
│  using whisperx.assign_word_speakers()                           │
└──────────────────────┬───────────────────────────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
       ┌────────────┐   ┌────────────┐
       │  .txt file  │   │ .json file │
       └────────────┘   └────────────┘
```

---

## 📊 Model Sizes & Performance

| Whisper Model | Parameters | VRAM (GPU) | Relative Speed | Quality |
|---|---|---|---|---|
| `tiny` | 39 M | ~1 GB | ★★★★★ | ★☆☆☆☆ |
| `base` | 74 M | ~1 GB | ★★★★☆ | ★★☆☆☆ |
| `small` | 244 M | ~2 GB | ★★★☆☆ | ★★★☆☆ |
| `medium` | 769 M | ~5 GB | ★★☆☆☆ | ★★★★☆ |
| `large-v2` | 1550 M | ~10 GB | ★☆☆☆☆ | ★★★★★ |
| `large-v3` | 1550 M | ~10 GB | ★☆☆☆☆ | ★★★★★ |

**Recommendations:**
- **Meeting / important recordings:** `large-v3` (best accuracy)
- **Quick transcription / testing:** `base` atau `small`
- **CPU-only / limited RAM:** `tiny` atau `base`

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|---|---|
| `ffmpeg not found` | Install ffmpeg (see above) |
| `401 Unauthorized` (pyannote) | Accept model conditions on HuggingFace + set valid HF_TOKEN |
| `CUDA out of memory` | Use a smaller model (`--model small`) or reduce `--batch-size 2` |
| `MPS not supported` | pyannote may not support MPS; pipeline auto-falls back to CPU for diarization |
| Slow on CPU | Expected — use GPU if available, or use `--model tiny/base` |
| Wrong language detected | Specify `--language xx` explicitly |

### Memory Tips

```bash
# If you hit OOM on GPU:
python pipeline.py audio.wav --model small --batch-size 2

# If you hit OOM on CPU:
python pipeline.py audio.wav --model tiny --batch-size 1
```

---

## 🔌 Extending the Pipeline

### Use as a Python Library

```python
from pipeline import transcribe, diarize, merge
from utils import preprocess_audio, save_outputs, get_device, get_compute_type

device = get_device()
compute_type = get_compute_type(device)

# Preprocess
wav_path = str(preprocess_audio("meeting.mp3"))

# Transcribe
result, audio = transcribe(wav_path, model_name="large-v3", device=device, compute_type=compute_type)

# Diarize
diarize_result = diarize(wav_path, hf_token="hf_XXXXX", device=device, num_speakers=3)

# Merge
segments = merge(result, diarize_result)

# Save
save_outputs(segments, "meeting_diarized")

# Or process segments programmatically
for seg in segments:
    print(f"{seg['speaker']}: {seg['text']}")
```

### Integration Ideas

- **Meeting transcription service** — Wrap pipeline in a FastAPI endpoint
- **Call recording analysis** — Batch-process call recordings for QA
- **Podcast transcription** — Generate show notes with speaker separation
- **Subtitle generation** — Convert word-level timestamps to SRT format
- **Real-time streaming** — Adapt for chunked audio processing

---

## 📜 License

This project uses open-source components:
- WhisperX — BSD License
- pyannote.audio — MIT License
- OpenAI Whisper — MIT License
- PyTorch — BSD License

Your usage must comply with each model's license and HuggingFace terms.
