import re
from collections import Counter
import math

# optional transformers
try:
    from transformers import pipeline, AutoTokenizer
    _HAS_TRANSFORMERS = True
except Exception:
    pipeline = None
    AutoTokenizer = None
    _HAS_TRANSFORMERS = False

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')

STOPWORDS = set([
    "the","and","is","in","it","of","to","a","we","that","this","for","on","with","as","are","be","by","an","at",
    "from","or","was","will","have","has","not","but","they","their","you","your","i","me","my"
])

def simple_sentence_tokenize(text):
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences

def score_sentences_by_freq(text):
    sentences = simple_sentence_tokenize(text)
    words = re.findall(r"\w+", text.lower())
    freqs = Counter(w for w in words if w not in STOPWORDS)
    if freqs:
        maxf = max(freqs.values())
    else:
        maxf = 1
    for w in list(freqs.keys()):
        freqs[w] = freqs[w] / float(maxf)
    sent_scores = []
    for s in sentences:
        ws = re.findall(r"\w+", s.lower())
        if not ws:
            continue
        score = sum(freqs.get(w, 0) for w in ws) / math.sqrt(len(ws))
        sent_scores.append((s, score))
    return sent_scores

def summarise_extractive(text, num_sentences=5):
    scored = score_sentences_by_freq(text)
    if not scored:
        return ""
    top = sorted(scored, key=lambda x: x[1], reverse=True)[:num_sentences]
    chosen = set(s for s, _ in top)
    sentences = simple_sentence_tokenize(text)
    out = [s for s in sentences if s in chosen]
    return " ".join(out)

class SummarizerError(Exception):
    pass

def _chunk_text_by_tokens(text, tokenizer, max_tokens=512, stride=50):
    """
    Chunk text approximately by tokens using tokenizer.encode to count tokens.
    Returns list of text chunks.
    """
    tokens = tokenizer.encode(text)
    chunks = []
    i = 0
    n = len(tokens)
    while i < n:
        j = min(i + max_tokens, n)
        chunk_tokens = tokens[i:j]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        i = j - stride  # overlap
    return chunks

def summarise_transformer(text, model_name="sshleifer/distilbart-cnn-12-6", max_chunk_tokens=512):
    """
    Chunk the input and summarize chunks with a transformer summarizer, then combine.
    Requires transformers to be installed.
    """
    if not _HAS_TRANSFORMERS:
        raise SummarizerError("transformers not installed. Install transformers to use transformer summarizer.")
    try:
        # instantiate a summarization pipeline (will download model if needed)
        summarizer = pipeline("summarization", model=model_name)
        # try to use tokenizer for better chunking
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            chunks = _chunk_text_by_tokens(text, tokenizer, max_tokens=max_chunk_tokens, stride=int(max_chunk_tokens * 0.2))
        except Exception:
            # fallback: naive character chunking
            chunk_size = 2000
            chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        partial_summaries = []
        for chunk in chunks:
            # avoid sending empty chunks
            if not chunk.strip():
                continue
            # summarizer returns list of dicts with 'summary_text'
            res = summarizer(chunk, max_length=200, min_length=30, do_sample=False)
            partial_summaries.append(res[0]["summary_text"])
        # combine partial summaries into final summary
        if not partial_summaries:
            return ""
        combined = " ".join(partial_summaries)
        # if combined is long, summarize again
        if len(combined.split()) > 300:
            res = summarizer(combined, max_length=200, min_length=50, do_sample=False)
            return res[0]["summary_text"]
        return combined
    except Exception as e:
        raise SummarizerError(f"transformer summarizer failed: {e}")

def extract_actions_and_decisions(full_text):
    sentences = simple_sentence_tokenize(full_text)
    actions = []
    decisions = []
    agendas = []
    action_keywords = ["todo", "action", "follow up", "follow-up", "will", "assign", "assigned", "deadline", "due"]
    decision_keywords = ["decision", "decided", "agreed", "approve", "approved", "resolution", "conclude"]
    agenda_keywords = ["agenda", "topic", "discuss", "discussed", "item"]
    for s in sentences:
        ls = s.lower()
        if any(k in ls for k in action_keywords):
            actions.append(s)
        if any(k in ls for k in decision_keywords):
            decisions.append(s)
        if any(k in ls for k in agenda_keywords):
            agendas.append(s)
    return agendas, decisions, actions

def build_structured_minutes(full_text, speakered_transcript):
    speakers = sorted({s["speaker"] for s in speakered_transcript})
    agendas, decisions, actions = extract_actions_and_decisions(full_text)
    exec_summary = summarise_extractive(full_text, num_sentences=4)
    long_summary = summarise_extractive(full_text, num_sentences=10)
    minutes = {
        "attendees": speakers,
        "agenda": agendas,
        "decisions": decisions,
        "todos": actions,
        "executive_summary": exec_summary,
        "long_summary": long_summary,
        "transcript": speakered_transcript,
    }
    return minutes
