import whisperx
print("Downloading whisperx model: large...")
model = whisperx.load_model("large", device="cpu", compute_type="int8")
print("Model 'large' downloaded and cached successfully!")
