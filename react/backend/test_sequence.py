import whisperx
import sys
import numpy as np

print("Loading whisper")
model = whisperx.load_model("large-v3", device="cpu", compute_type="float32")

# Create a dummy audio array
audio = np.zeros(16000 * 5, dtype=np.float32)

print("Transcribing")
result = model.transcribe(audio, batch_size=4)
print("Detected language:", result.get("language", "unknown"))

print("Loading align model")
try:
    align_model, align_meta = whisperx.load_align_model(language_code="en", device="cpu")
    print("Align model loaded successfully")
except Exception as e:
    print(f"Exception caught: {e}")

print("Done")
