from textblob import TextBlob

def offline_summarize(text: str, tone="neutral", style="simple", length="medium") -> str:
    """
    Simple offline summarizer using TextBlob.
    Respects tone, style, and length.
    """
    blob = TextBlob(text)
    sentences = blob.sentences

    if not sentences:
        return "No content to summarize."

    # Control length
    if length == "short":
        n = min(2, len(sentences))
    elif length == "long":
        n = min(6, len(sentences))
    else:  # medium
        n = min(4, len(sentences))

    summary = " ".join(str(s) for s in sentences[:n])

    # Apply style
    if style == "bullet":
        summary = " • " + "\n • ".join(str(s) for s in sentences[:n])
    elif style == "headline":
        summary = summary.split(".")[0] + "."

    # Apply tone
    if tone == "formal":
        summary = summary.replace("!", ".")
    elif tone == "casual":
        summary = "👉 " + summary

    return summary
