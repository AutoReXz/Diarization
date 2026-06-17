import whisperx
import sys
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    print("Loading whisper")
    compute_type = "float16" if device == "cuda" else "float32"
    model = whisperx.load_model("base", device=device, compute_type=compute_type)

    print("Loading audio")
    # Provide a valid audio file path from the project
    audio_file = "c:/Nitro/belajar/Diarization/react/backend/test.wav"
    import numpy as np
    audio = np.zeros(16000 * 5, dtype=np.float32)

    print("Transcribing")
    result = model.transcribe(audio, batch_size=4)
    print("Detected language:", result.get("language", "unknown"))

    print("Loading align model")
    align_model, align_meta = whisperx.load_align_model(language_code="en", device=device)
    print("Align model loaded successfully")
except Exception as e:
    import traceback
    traceback.print_exc()

print("Done")
