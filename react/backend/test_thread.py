import whisperx
import numpy as np
import concurrent.futures

def run_pipeline():
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

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_pipeline)
        future.result()
