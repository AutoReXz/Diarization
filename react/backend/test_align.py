import whisperx
import sys

print("Loading whisperx")
try:
    print("Loading align model...")
    align_model, align_meta = whisperx.load_align_model(language_code="en", device="cpu")
    print("Align model loaded successfully")
except Exception as e:
    print(f"Exception caught: {e}")

print("Done")
