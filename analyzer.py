# analyzer.py
"""
Phase 2 analyzer: summarization (short + bullets), sentiment, keywords, NER, simple stats.
Relies on extractor.extract(url) from Phase 1.
"""
from typing import Dict, List
from transformers import pipeline
import re
import math

# Import the extractor you already have
from extractor import extract, _clean_text  # extractor.py from Phase1

# NLP models (light/fast choices)
# Summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Tone rewrite (optional; used if available)
try:
    rewriter = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
except Exception:
    rewriter = None

# Sentiment classifier
sentiment_clf = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# spaCy for NER (you must install `spacy` and download model en_core_web_sm)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    _HAS_SPACY = True
except Exception:
    nlp = None
    _HAS_SPACY = False

# sklearn for TF-IDF keywords
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _HAS_SK = True
except Exception:
    _HAS_SK = False


def chunk_text_simple(text: str, max_chars: int = 3000):
    text = _clean_text(text)
    chunks = []
    while len(text) > max_chars:
        cut = text.rfind('.', 0, max_chars)
        if cut == -1 or cut < int(max_chars * 0.6):
            cut = max_chars
        chunks.append(text[:cut+1])
        text = text[cut+1:]
    if text:
        chunks.append(text)
    return chunks


def summarize_plain(text: str, max_len=180, min_len=30) -> str:
    if not text or len(text.split()) < 30:
        return _clean_text(text or "")
    parts = []
    for c in chunk_text_simple(text, max_chars=2500):
        out = summarizer(c, max_length=max_len, min_length=min_len, do_sample=False)
        parts.append(out[0]["summary_text"].strip())
    combined = " ".join(parts)
    # If super long, summarize again
    if len(combined) > 2000:
        combined = summarizer(combined, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
    return _clean_text(combined)


def rewrite_tone(summary: str, mode: str) -> str:
    mode = (mode or "").lower()
    if not summary:
        return ""
    if rewriter:
        if "fact" in mode:
            prompt = "Rewrite only with facts. Produce 5-8 bullet points:\n\n" + summary
        elif "10" in mode or "child" in mode or "explain" in mode:
            prompt = "Explain to a 10-year-old using short simple sentences:\n\n" + summary
        else:
            prompt = "Rewrite neutrally in a journalistic tone, concise 4-6 sentences:\n\n" + summary
        try:
            out = rewriter(prompt, num_return_sequences=1)[0]["generated_text"]
            return _clean_text(out)
        except Exception:
            pass
    # fallback
    if "fact" in mode:
        sents = re.split(r'(?<=[.!?])\s+', summary)
        bullets = [f"- {s.strip()}" for s in sents if len(s.split()) >= 3]
        return "\n".join(bullets[:8])
    if "10" in mode or "child" in mode:
        return "For a 10-year-old: " + summary
    return summary


def get_sentiment(text: str) -> Dict:
    if not text:
        return {"label": "NEUTRAL", "score": 0.0}
    res = sentiment_clf(text[:4000])[0]
    return {"label": res["label"], "score": float(res["score"])}


def extract_keywords(text: str, top_n: int = 8) -> List[str]:
    if not _HAS_SK or not text or len(text.split()) < 20:
        # fallback: simple freq-based keywords
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [w for w, _ in top]
    # TF-IDF on single doc: create tiny corpus by splitting into pseudo-sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) < 2:
        corpus = [text]
    else:
        corpus = sentences
    vect = TfidfVectorizer(stop_words="english", max_features=500)
    X = vect.fit_transform(corpus)
    feature_names = vect.get_feature_names_out()
    # compute mean tfidf across pseudo-sentences and pick top features
    import numpy as np
    scores = X.mean(axis=0).A1
    ranked_idx = scores.argsort()[::-1][:top_n]
    return [feature_names[i] for i in ranked_idx if scores[i] > 0][:top_n]


def extract_entities(text: str) -> List[Dict]:
    if not _HAS_SPACY or not text:
        return []
    doc = nlp(text)
    ents = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return ents


def reading_time_and_counts(text: str) -> Dict:
    words = len(text.split())
    # Avg reading speed 200 wpm
    minutes = words / 200.0
    return {"words": words, "reading_time_min": round(minutes, 2)}


def analyze_url(url: str, tone_mode: str = "neutral") -> Dict:
    out = extract(url)
    if out["status"] == "fail":
        return {"error": "extractor_failed", "method": out.get("method", "none"), "meta": out.get("meta", {})}
    text = out.get("text", "")
    meta = out.get("meta", {})
    summary_short = summarize_plain(text, max_len=120, min_len=20)
    summary_long = summarize_plain(text, max_len=220, min_len=60)
    summary_tone = rewrite_tone(summary_short, tone_mode)
    bullets = rewrite_tone(summary_long, "fact-only") if "fact" in tone_mode else None
    sentiment = get_sentiment(text)
    keywords = extract_keywords(text, top_n=8)
    entities = extract_entities(text)
    stats = reading_time_and_counts(text)
    return {
        "method": out.get("method"),
        "meta": meta,
        "summary_short": summary_short,
        "summary_tone": summary_tone,
        "summary_bullets": bullets,
        "sentiment": sentiment,
        "keywords": keywords,
        "entities": entities,
        "stats": stats,
        "full_text": text
    }
