````markdown
# Local Meeting Minutes (whisper + optional diarization + summarizer)

This project provides a small local web app to transcribe meeting audio (mp3/m4a/wav) using Whisper, perform speaker diarization (multiple backends), and produce structured minutes (agenda / decisions / todos) and summaries.

What's new in this change
- Optional diarization backends:
  - resemblyzer (lightweight, default)
  - whisperx (optional; integrates with Whisper alignment)
  - pyannote (optional; higher-accuracy, requires Hugging Face token)
- Automatic speaker-count estimation using HDBSCAN (optional)
- Optional transformer-based summarizer (chunked) using Hugging Face transformers

Quick setup
1. Install system deps:
   - ffmpeg
2. Create venv:
   python -m venv .venv
   source .venv/bin/activate
3. Install core Python deps:
   pip install -r requirements.txt

Note on optional packages
- whisperx: install via its repo if you want whisperx diarization:
  pip install git+https://github.com/m-bain/whisperX.git
- pyannote.audio: install separately and set your Hugging Face token in HF_TOKEN or HUGGINGFACE_TOKEN environment variable:
  pip install pyannote.audio
  export HF_TOKEN="your_hf_token"
- transformers: for transformer summarization:
  pip install transformers
- hdbscan: for speaker-count estimation:
  pip install hdbscan

Run the app
export FLASK_APP=app.py
flask run
Open http://127.0.0.1:5000

Web UI options
- Whisper model: choose model size (tiny/base/small/medium/large)
- Diarization method: resemblyzer (default), whisperx, pyannote
- Auto-estimate speaker count: try automatic estimation using HDBSCAN (if installed)
- Summarizer: Extractive (default) or Transformer (requires transformers)
- Transformer model: change summarizer model (default sshleifer/distilbart-cnn-12-6)

Notes and caveats
- Optional dependencies (whisperx, pyannote, hdbscan, transformers) are large and may require significant disk/CPU/GPU resources.
- pyannote usage requires a Hugging Face token (HF_TOKEN) because some pretrained models require authentication.
- If optional libraries are missing, the app will fallback to the lightweight resemblyzer path when possible and provide warnings.
- For production use and better diarization performance, consider running pyannote with GPU and a proper Hugging Face token.

If you'd like, I can:
- Create a branch and open a pull request with these files (I couldn't push yet — your repo appears empty or inaccessible from my side).
- Generate a patch file for easier application.
````
