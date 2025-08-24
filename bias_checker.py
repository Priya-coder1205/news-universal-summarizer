# bias_checker.py
from __future__ import annotations
from typing import List, Dict
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import streamlit as st

# Make sure VADER lexicon is available
nltk.download("vader_lexicon", quiet=True)
vader = SentimentIntensityAnalyzer()

def sentiment_label(polarity: float) -> str:
    if polarity >= 0.2:
        return "Positive"
    elif polarity <= -0.2:
        return "Negative"
    else:
        return "Neutral"

# -------------------------------
# Sentiment + Bias Analysis
# -------------------------------

def analyze(text: str) -> Dict:
    """
    Analyze a single text.
    Returns polarity (-1..1), subjectivity (0..1),
    sentiment label, and flagged emotive/bias words.
    """
    try:
        blob = TextBlob(text)
        p = float(blob.sentiment.polarity)
        s = float(blob.sentiment.subjectivity)
    except Exception:
        p, s = 0.0, 0.0

    # Assign label based on polarity
    label = sentiment_label(p)

    # Flag emotive/bias words using VADER
    flagged = []
    for tok in text.split():
        score = vader.polarity_scores(tok)["compound"]
        if abs(score) >= 0.5:  # strong emotional tone
            flagged.append(tok.lower())

    # Remove duplicates but preserve order
    seen = set()
    flagged_unique = [x for x in flagged if not (x in seen or seen.add(x))]

    return {
        "polarity": p,
        "subjectivity": s,
        "label": label,
        "flagged_words": flagged_unique
    }

def compare(sources: List[Dict]) -> Dict:
    """
    Compare sentiment across multiple sources.
    Returns bias flag, explanatory message, spread,
    and per-source analysis.
    """
    vals = []
    for it in sources:
        t = (it.get("text") or "").strip()
        if not t:
            continue
        a = analyze(t)
        vals.append({
            "title": it.get("title",""),
            "url": it.get("url",""),
            **a
        })

    if not vals:
        return {"flag": False, "message": "No text available to analyze.", "details": []}

    # Compute spread in polarity across sources
    polarities = [v["polarity"] for v in vals]
    p_min, p_max = min(polarities), max(polarities)
    spread = p_max - p_min

    # Improved threshold logic
    if spread >= 0.6:
        flag = True
        msg = f"⚠️ Potential bias detected: sources diverge strongly (spread {spread:.2f})."
    elif spread >= 0.3:
        flag = True
        msg = f"⚠️ Mild bias possible: some variation in sentiment (spread {spread:.2f})."
    else:
        flag = False
        msg = f"✅ Sources show very similar sentiment (spread {spread:.2f}) — broad agreement."

    return {"flag": flag, "message": msg, "details": vals, "spread": spread}

# -------------------------------
# Visualization in Streamlit
# -------------------------------

def visualize_sentiment_table(details: List[Dict]):
    """
    Show a color-coded sentiment/bias comparison table
    and a polarity bar chart.
    """
    if not details:
        st.info("No sentiment details available.")
        return

    df = pd.DataFrame(details)[["title", "polarity", "subjectivity", "label", "flagged_words"]]

    # Coloring functions
    def color_sentiment(val):
        if val > 0.2:
            return "background-color: #b3e6ff"  # light blue = positive
        elif val < -0.2:
            return "background-color: #ffb3b3"  # light red = negative
        else:
            return "background-color: #b3ffb3"  # light green = neutral

    def color_subjectivity(val):
        if val > 0.55:
            return "background-color: #fff3b3"  # yellow = opinion-heavy
        return ""

    styled = df.style.applymap(color_sentiment, subset=["polarity"]) \
                     .applymap(color_subjectivity, subset=["subjectivity"])

    st.markdown("### 📊 Sentiment Comparison Table")
    st.dataframe(styled, use_container_width=True)

    st.markdown("### 📈 Sentiment Spread Across Sources")
    st.bar_chart(df.set_index("title")["polarity"])

