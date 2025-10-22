"""
Diarization utilities with multiple backend support.

Provides:
- diarize_file(path, method='resemblyzer'|'whisperx'|'pyannote', num_speakers=None)
- estimate_num_speakers(path) -> int

All optional heavy dependencies are imported only when needed and raise informative errors.
"""
import os
import tempfile
from typing import List, Dict

# Common lightweight deps
import numpy as np
import soundfile as sf

# resemblyzer embedding/clustering fallback
try:
    from resemblyzer import VoiceEncoder
    _HAS_RESEMBLYZER = True
except Exception:
    VoiceEncoder = None
    _HAS_RESEMBLYZER = False

# sklearn clustering
try:
    from sklearn.cluster import AgglomerativeClustering
    _HAS_SKLEARN = True
except Exception:
    AgglomerativeClustering = None
    _HAS_SKLEARN = False

# optional HDBSCAN for automatic estimation
try:
    import hdbscan
    _HAS_HDBSCAN = True
except Exception:
    hdbscan = None
    _HAS_HDBSCAN = False

# optional whisperx (very optional)
try:
    import whisperx
    _HAS_WHISPERX = True
except Exception:
    whisperx = None
    _HAS_WHISPERX = False

# optional pyannote
try:
    from pyannote.audio import Pipeline as PyannotePipeline
    _HAS_PYANNOTE = True
except Exception:
    PyannotePipeline = None
    _HAS_PYANNOTE = False

# helper
class DiarizationError(Exception):
    pass

def ensure_wav(path: str) -> str:
    """
    Convert non-wav file to a temporary wav at 16k mono PCM.
    Requires ffmpeg CLI available in PATH.
    """
    base, ext = os.path.splitext(path)
    if ext.lower() == ".wav":
        return path
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    # lazy import so module doesn't require ffmpeg if user doesn't use it
    import subprocess
    subprocess.check_call(
        ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", "-vn", "-acodec", "pcm_s16le", tmp_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return tmp_path

def sliding_window(wav, sr, window_size=1.5, hop=0.75):
    win_samps = int(window_size * sr)
    hop_samps = int(hop * sr)
    total = len(wav)
    pos = 0
    while pos < total:
        end = min(pos + win_samps, total)
        chunk = wav[pos:end]
        start_time = pos / sr
        end_time = end / sr
        yield start_time, end_time, chunk
        pos += hop_samps

def diarize_resemblyzer(path: str, num_speakers: int = 2) -> List[Dict]:
    if not _HAS_RESEMBLYZER:
        raise DiarizationError("resemblyzer is not installed. Install resemblyzer to use this method.")
    if not _HAS_SKLEARN:
        raise DiarizationError("scikit-learn is required for clustering. Install scikit-learn.")
    wav_path = ensure_wav(path)
    wav, sr = sf.read(wav_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if np.max(np.abs(wav)) > 0:
        wav = wav / np.max(np.abs(wav))
    encoder = VoiceEncoder()
    starts, ends, embeddings = [], [], []
    for start, end, chunk in sliding_window(wav, sr, window_size=1.5, hop=0.75):
        if len(chunk) < 0.2 * sr:
            continue
        emb = encoder.embed_utterance(chunk, rate=sr)
        starts.append(start)
        ends.append(end)
        embeddings.append(emb)
    if not embeddings:
        return []
    X = np.vstack(embeddings)
    n = max(2, int(num_speakers or 2))
    clustering = AgglomerativeClustering(n_clusters=n).fit(X)
    labels = clustering.labels_
    # merge contiguous windows with same label
    segments = []
    cur_label = labels[0]
    cur_start = starts[0]
    cur_end = ends[0]
    for i in range(1, len(labels)):
        if labels[i] == cur_label and abs(starts[i] - cur_end) <= 0.5:
            cur_end = ends[i]
        else:
            segments.append({"start": float(cur_start), "end": float(cur_end), "speaker": f"Speaker {int(cur_label)+1}"})
            cur_label = labels[i]
            cur_start = starts[i]
            cur_end = ends[i]
    segments.append({"start": float(cur_start), "end": float(cur_end), "speaker": f"Speaker {int(cur_label)+1}"})
    if wav_path != path:
        try:
            os.remove(wav_path)
        except Exception:
            pass
    return segments

def diarize_whisperx(path: str, num_speakers: int = None) -> List[Dict]:
    if not _HAS_WHISPERX:
        raise DiarizationError("whisperx is not installed. Install whisperx to use this method.")
    # whisperx expects whisper model and audio path. We'll run the whisperx pipeline to get segments + diarization
    try:
        # whisperx requires a whisper model object; the main app loads whisper model so we import again here
        import whisper
        model = whisper.load_model("base")  # small default; main app uses selected model for transcription
        result = model.transcribe(path, verbose=False)
        # align and diarize using whisperx
        # Note: whisperx APIs may evolve; this is a best-effort wrapper
        from whisperx import load_align_model, diarize
        # align model
        device = "cuda" if whisperx.is_cuda_available() else "cpu"
        align_model, metadata = load_align_model(language_code=result["language"], device=device)
        # diarize returns speaker segments when given audio path
        diarize_segments = diarize(path, result, device=device)  # whisperx.diarize wrapper
        # diarize_segments is list of dicts {"start":..,"end":..,"speaker":..}
        return [{"start": float(s["start"]), "end": float(s["end"]), "speaker": f"Speaker {s.get('speaker', 0)}"} for s in diarize_segments]
    except Exception as e:
        raise DiarizationError(f"whisperx diarization failed: {e}")

def diarize_pyannote(path: str, num_speakers: int = None) -> List[Dict]:
    if not _HAS_PYANNOTE:
        raise DiarizationError("pyannote.audio is not installed. Install pyannote.audio to use this method.")
    # pyannote requires a HF token in environment: set PYANNOTE or HF_TOKEN accordingly
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise DiarizationError("pyannote usage requires a Hugging Face token in HF_TOKEN or HUGGINGFACE_TOKEN environment variable.")
    try:
        pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
        diarization = pipeline(path)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": f"Speaker {speaker}"})
        return segments
    except Exception as e:
        raise DiarizationError(f"pyannote diarization failed: {e}")

def estimate_num_speakers(path: str) -> int:
    """
    Estimate number of speakers using HDBSCAN on resemblyzer embeddings.
    """
    if not _HAS_RESEMBLYZER:
        raise RuntimeError("resemblyzer is required for speaker estimation (install resemblyzer).")
    wav_path = ensure_wav(path)
    wav, sr = sf.read(wav_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if np.max(np.abs(wav)) > 0:
        wav = wav / np.max(np.abs(wav))
    encoder = VoiceEncoder()
    embeddings = []
    for start, end, chunk in sliding_window(wav, sr, window_size=1.5, hop=0.75):
        if len(chunk) < 0.2 * sr:
            continue
        emb = encoder.embed_utterance(chunk, rate=sr)
        embeddings.append(emb)
    if not embeddings:
        return 0
    X = np.vstack(embeddings)
    if _HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        labels = clusterer.fit_predict(X)
        unique = set([l for l in labels if l >= 0])
        return max(2, len(unique))
    else:
        # fallback heuristic: use Agglomerative clustering with 2..5 and pick elbow by silhouette-like heuristic
        if not _HAS_SKLEARN:
            return 2
        from sklearn.metrics import pairwise_distances
        best_k = 2
        last_score = None
        for k in range(2, min(6, len(X))):
            clustering = AgglomerativeClustering(n_clusters=k).fit(X)
            labels = clustering.labels_
            # simple score: average intra-cluster distance (lower is better)
            dists = pairwise_distances(X)
            intra = 0.0
            count = 0
            for i in range(len(X)):
                for j in range(i+1, len(X)):
                    if labels[i] == labels[j]:
                        intra += dists[i, j]
                        count += 1
            score = intra / max(1, count)
            if last_score is None or score < last_score:
                best_k = k
                last_score = score
        return best_k

def diarize_file(path: str, method: str = "resemblyzer", num_speakers: int = None) -> List[Dict]:
    method = (method or "resemblyzer").lower()
    if method == "resemblyzer":
        return diarize_resemblyzer(path, num_speakers or 2)
    elif method == "whisperx":
        return diarize_whisperx(path, num_speakers)
    elif method == "pyannote":
        return diarize_pyannote(path, num_speakers)
    else:
        raise DiarizationError(f"Unknown diarization method: {method}")
