#!/usr/bin/env python3
import os
import tempfile
import math
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Local modules
from diarize import diarize_file, estimate_num_speakers, DiarizationError
from summarizer import (
    build_structured_minutes,
    summarise_extractive,
    summarise_transformer,
    SummarizerError,
)

# whisper main import (we use whisper or whisper.load_model dynamically)
try:
    import whisper
    _HAS_WHISPER = True
except Exception:
    whisper = None
    _HAS_WHISPER = False

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

WHISPER_MODELS = {}

def get_whisper_model(name="base"):
    if not _HAS_WHISPER:
        raise RuntimeError("whisper package is not installed.")
    if name in WHISPER_MODELS:
        return WHISPER_MODELS[name]
    app.logger.info(f"Loading Whisper model: {name} ...")
    model = whisper.load_model(name)
    WHISPER_MODELS[name] = model
    return model

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, warnings=[]) 

@app.route("/transcribe", methods=["POST"])
def transcribe():
    warnings = []
    if "audio" not in request.files:
        flash("No audio file provided", "error")
        return redirect(url_for("index"))
    file = request.files["audio"]
    if file.filename == "":
        flash("No audio file selected", "error")
        return redirect(url_for("index"))
    filename = secure_filename(file.filename)
    local_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(local_path)

    model_name = request.form.get("model") or "base"
    diarize_method = request.form.get("diarize_method") or "resemblyzer"
    summarizer_choice = request.form.get("summarizer") or "extractive"
    transformer_model = request.form.get("transformer_model") or "sshleifer/distilbart-cnn-12-6"
    auto_estimate = request.form.get("auto_estimate") == "on"

    try:
        whisper_model = get_whisper_model(model_name)
    except Exception as e:
        # fallback: try to import but do not fail completely
        whisper_model = None
        warnings.append(f"Whisper not available: {e}. Transcription will fail if whisper is required.")

    # 1) Transcribe with whisper (if available)
    transcript_segments = []
    full_text = ""
    if whisper_model is not None:
        # keep timestamps for segments
        try:
            result = whisper_model.transcribe(local_path, verbose=False)
            for seg in result.get("segments", []):
                transcript_segments.append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})
            full_text = " ".join(s["text"] for s in transcript_segments)
        except Exception as e:
            warnings.append(f"Whisper transcription failed: {e}")
    else:
        warnings.append("Whisper model not loaded. Cannot transcribe audio. Please install whisper and restart.")
        # return page with warning
        return render_template("index.html", result=None, warnings=warnings)

    # 2) Optionally estimate num_speakers
    num_speakers_provided = None
    try:
        ns = int(request.form.get("num_speakers") or 0)
        if ns > 0:
            num_speakers_provided = ns
    except ValueError:
        num_speakers_provided = None

    estimated_speakers = None
    if auto_estimate and num_speakers_provided is None:
        try:
            estimated_speakers = estimate_num_speakers(local_path)
            if estimated_speakers and estimated_speakers > 0:
                num_speakers_provided = estimated_speakers
        except Exception as e:
            warnings.append(f"Automatic speaker estimation failed: {e}")

    # 3) Diarize using chosen method
    try:
        speaker_segments = diarize_file(
            local_path, method=diarize_method, num_speakers=num_speakers_provided
        )
    except DiarizationError as e:
        warnings.append(str(e))
        # fallback to empty segments (map everything to Speaker 1)
        speaker_segments = []

    # 4) Map speakers to transcript segments (simple overlap heuristic)
    def map_speakers_to_segments(transcript_segments, speaker_segments):
        out = []
        for seg in transcript_segments:
            start = seg["start"]
            end = seg["end"]
            text = seg["text"].strip()
            if not text:
                continue
            overlaps = {}
            for s in speaker_segments:
                overlap_start = max(start, s["start"])
                overlap_end = min(end, s["end"])
                if overlap_end > overlap_start:
                    overlaps.setdefault(s["speaker"], 0)
                    overlaps[s["speaker"]] += overlap_end - overlap_start
            speaker = max(overlaps.items(), key=lambda x: x[1])[0] if overlaps else "Speaker 1"
            out.append({"speaker": speaker, "start": start, "end": end, "text": text})
        return out

    speakered_transcript = map_speakers_to_segments(transcript_segments, speaker_segments)
    if not speakered_transcript:
        # fallback: single speaker using whole text
        speakered_transcript = [{"speaker": "Speaker 1", "start": 0.0, "end": transcript_segments[-1]["end"] if transcript_segments else 0.0, "text": full_text}]

    # 5) Build structured minutes
    minutes = build_structured_minutes(full_text, speakered_transcript)

    # 6) Summaries
    exec_summary = summarise_extractive(full_text, num_sentences=5)
    long_summary = summarise_extractive(full_text, num_sentences=12)
    transformer_summary = None

    if summarizer_choice == "transformer":
        try:
            transformer_summary = summarise_transformer(full_text, model_name=transformer_model)
        except SummarizerError as e:
            warnings.append(f"Transformer summarizer failed: {e}")
            transformer_summary = None

    return render_template(
        "index.html",
        result={
            "filename": filename,
            "speakered_transcript": speakered_transcript,
            "speaker_segments": speaker_segments,
            "minutes": minutes,
            "executive_summary": exec_summary,
            "long_summary": long_summary,
            "transformer_summary": transformer_summary,
        },
        warnings=warnings,
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
