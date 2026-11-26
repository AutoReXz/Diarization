# Speech Transcription and Speaker Diarization

A local pipeline that transcribes audio and labels segments by speaker. WhisperX handles transcription and word alignment, pyannote identifies speaker turns, and ffmpeg prepares the input. The pipeline exports a plain-text transcript and structured JSON.

## Run Locally

Install ffmpeg and Git, then create a Python 3.11 virtual environment:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS or Linux, or `\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The requirements include PyTorch, torchaudio, and torchvision. Use compatible builds for your operating system and accelerator. The first run downloads model artifacts. Accept the access conditions for the diarization model and set `HF_TOKEN` in the shell that runs the pipeline.

[PyTorch installation](https://pytorch.org/get-started/locally/) · [pyannote Community-1 model](https://huggingface.co/pyannote/speaker-diarization-community-1) · [Hugging Face token settings](https://huggingface.co/settings/tokens)

## Transcribe an Audio File

```bash
python pipeline.py meeting.wav --model base --language en --output-dir results
```

Omit `--language` for language detection. If you know the speaker count, pass `--num-speakers`. Run `python pipeline.py --help` for all options.

The example writes `meeting_diarized.txt` and `meeting_diarized.json` to `results/`. See the [example output files](examples/) for timestamp, text, and speaker label formats.

## Interfaces and Source

- [pipeline.py](pipeline.py) — Command-line transcription, alignment, diarization, and export.
- [app.py](app.py) — Local Streamlit interface. Start it with `python -m streamlit run app.py`.
- [react/](react/) — React interface and its Python backend, as an alternate interface.
- [requirements.txt](requirements.txt) — Python dependencies for the local pipeline and Streamlit app.

## Troubleshooting

If the process runs out of memory, try a smaller model with `--model base` or reduce `--batch-size`. Verify that ffmpeg is on your PATH if audio preprocessing fails.
