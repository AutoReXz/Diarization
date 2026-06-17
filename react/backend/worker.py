import os
import sys
import json
import time
import traceback

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def send_event(event_type: str, data: dict):
    # Print JSON line for the parent process to read
    print(json.dumps({"event": event_type, "data": data}), flush=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: worker.py <job_json_path>")
        sys.exit(1)
        
    job_path = sys.argv[1]
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)
        
    audio_path = job["audio_path"]
    hf_token = job["hf_token"]
    model_name = job["model_name"]
    device = job["device"]
    compute_type = job["compute_type"]
    batch_size = job["batch_size"]
    lang_val = job["language"]
    
    num_speakers = job.get("num_speakers")
    min_speakers = job.get("min_speakers")
    max_speakers = job.get("max_speakers")

    import whisperx
    import torch
    import pandas as pd

    try:
        timings = {}

        # Step 1: Transcribe
        send_event("step", {
            "step": 1, "status": "active", "title": "Loading ASR model and transcribing"
        })

        t0 = time.time()
        bs = min(batch_size, 4) if device == "cpu" else batch_size
        model = whisperx.load_model(model_name, device=device, compute_type=compute_type, language=lang_val)
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=bs)
        detected_lang = result.get("language", lang_val or "unknown")
        timings["transcription"] = time.time() - t0

        send_event("step", {
            "step": 1, "status": "done",
            "title": "Transcription complete",
            "detail": f"Language: {detected_lang} / Segments: {len(result['segments'])} / {timings['transcription']:.1f}s"
        })

        # Free memory of whisper model before loading alignment model
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 2: Alignment
        send_event("step", {
            "step": 2, "status": "active", "title": "Running forced alignment"
        })

        t0 = time.time()
        align_model, align_meta = whisperx.load_align_model(language_code=detected_lang, device=device)
        result = whisperx.align(
            result["segments"], align_model, align_meta, audio,
            device=device, return_char_alignments=False
        )
        timings["alignment"] = time.time() - t0

        send_event("step", {
            "step": 2, "status": "done",
            "title": "Alignment complete",
            "detail": f"Word-level timestamps generated / {timings['alignment']:.1f}s"
        })

        del align_model
        del align_meta
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 3: Diarization
        send_event("step", {
            "step": 3, "status": "active", "title": "Running speaker diarization"
        })

        t0 = time.time()
        from pyannote.audio import Pipeline as PyannotePipeline
        diarize_pipe = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=hf_token
        )
        if not diarize_pipe:
            raise ValueError("Invalid HF token or missing pyannote model access")
        
        if device == "cuda":
            diarize_pipe.to(torch.device("cuda"))
        elif device == "mps":
            try:
                diarize_pipe.to(torch.device("mps"))
            except Exception:
                pass

        diarize_kwargs = {}
        if num_speakers and num_speakers > 0:
            diarize_kwargs["num_speakers"] = num_speakers
        if min_speakers and min_speakers > 0:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers and max_speakers > 0:
            diarize_kwargs["max_speakers"] = max_speakers

        waveform = torch.from_numpy(audio).unsqueeze(0)
        audio_input = {"waveform": waveform, "sample_rate": 16000}
        
        diarize_result = diarize_pipe(audio_input, **diarize_kwargs)

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
            for turn, _, speaker in annotation:
                diarize_list.append({"start": turn.start, "end": turn.end, "speaker": speaker})

        diarize_df = pd.DataFrame(diarize_list)
        timings["diarization"] = time.time() - t0

        send_event("step", {
            "step": 3, "status": "done",
            "title": "Diarization complete",
            "detail": f"{timings['diarization']:.1f}s"
        })

        del diarize_pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 4: Merge
        send_event("step", {
            "step": 4, "status": "active", "title": "Merging transcription with speaker labels"
        })

        t0 = time.time()
        merged = whisperx.assign_word_speakers(diarize_df, result)

        output_segments = []
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

        speaker_map = {}
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

        send_event("step", {
            "step": 4, "status": "done",
            "title": "Merge complete",
            "detail": f"Speakers: {len(speaker_map)} / Segments: {len(output_segments)} / {timings['merge']:.1f}s"
        })

        # Save final results to disk
        out_path = job_path + ".result.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"segments": output_segments, "timings": timings}, f)

        send_event("complete", {
            "segments": output_segments,
            "timings": timings,
        })

    except Exception as e:
        traceback.print_exc()
        send_event("error", {"detail": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    main()
