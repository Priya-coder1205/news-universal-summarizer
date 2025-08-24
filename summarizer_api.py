# summarizer_api.py
import os
from typing import Literal, Tuple, Optional

from textblob import TextBlob

def analyze_bias(text: str):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    
    if sentiment > 0.2:
        bias_flag = "Positive leaning"
    elif sentiment < -0.2:
        bias_flag = "Negative leaning"
    else:
        bias_flag = "Neutral / Balanced"
    
    return {"sentiment": sentiment, "bias_flag": bias_flag}

# --- Offline fallback (simple, deterministic) ---
def _offline_summarize(text: str, tone: str = "neutral", length: str = "medium", style: str = "simple") -> str:
    """
    Lightweight offline summary:
    - length: short (2 sentences), medium (4), long (6)
    - tone: neutral | fact-only | kid-friendly
    """
    import re
    # naive sentence split, but stable
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return "No content to summarize."

    n = 4
    if length == "short":
        n = 2
    elif length == "long":
        n = 6

    base = " ".join(sentences[:min(n, len(sentences))])

    if tone == "fact-only":
        # remove subjective phrases (very rough heuristic)
        base = re.sub(r"\b(I think|we believe|likely|apparently|reportedly|seems|could|might|maybe)\b", "", base, flags=re.I)
    elif tone == "kid-friendly":
        base = "Explain like I’m 10: " + base

    return base.strip()

# --- Providers (optional) ---
# OpenAI
_openai_client = None
try:
    from openai import OpenAI
    if os.getenv("OPENAI_API_KEY"):
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    _openai_client = None

# Gemini
_gemini_model = None
try:
    import google.generativeai as genai
    if os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash")
except Exception:
    _gemini_model = None

Tone = Literal["neutral", "fact-only", "kid-friendly"]
Length = Literal["short", "medium", "long"]

def _build_prompt(text: str, tone: Tone, length: Length, style: str) -> str:
    print("DEBUG _build_prompt → tone:", tone)

    target = {"short": "~80 words", "medium": "~160 words", "long": "~260 words"}[length]

    guidelines_map = {
        "neutral": "Write objectively. No speculation. Avoid emotional words or adjectives.",
        "fact-only": "ONLY include verifiable facts from the article. REMOVE all opinions, speculation, emotions, or descriptive adjectives.",
        "kid-friendly": "EXPLAIN like to a 10-year-old. Use short, simple sentences and very easy words. No jargon. Use examples a kid can understand.",
    }

    guidelines = guidelines_map[tone]

    return (
        f"You are a precise news summarizer.\n"
        f"- Writing style: {style}\n"
        f"- Target length: {target}\n"
        f"- TONE INSTRUCTIONS (VERY IMPORTANT): {guidelines}\n\n"
        f"Now summarize the following article:\n\n"
        f"ARTICLE START\n{text}\nARTICLE END\n\n"
        f"Remember: strictly follow the tone instructions above."
    )

def _try_openai(text: str, tone: Tone, length: Length, style: str) -> Optional[str]:
    if not _openai_client:
        return None
    try:
        prompt = _build_prompt(text, tone, length, style)
        # 🔍 DEBUG: print the prompt before sending
        print("\n--- OPENAI PROMPT ---")
        print(prompt)
        print("---------------------\n")
        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("OpenAI failed:", e)
        return None

def _try_gemini(text: str, tone: Tone, length: Length, style: str) -> Optional[str]:
    if not _gemini_model:
        return None
    try:
        prompt = _build_prompt(text, tone, length, style)
        # 🔍 DEBUG: print the prompt before sending
        print("\n--- GEMINI PROMPT ---")
        print(prompt)
        print("---------------------\n")
        resp = _gemini_model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        print("Gemini failed:", e)
        return None

def get_summary(
    text: str,
    tone: str = "neutral",          # 👈 use plain str instead of Tone
    length: str = "medium",         # 👈 use plain str instead of Length
    style: str = "simple",
    provider_order: tuple = ("openai", "gemini", "offline"),
) -> str:
    """
    Try summarization in order of providers.
    Falls back automatically if one fails.
    """
    for provider in provider_order:
        print(f"[DEBUG] Trying provider: {provider}")
        try:
            if provider == "openai":
                out = _try_openai(text, tone, length, style)
                if out:
                    print("[DEBUG] ✅ Used OpenAI")
                    return out

            elif provider == "gemini":
                out = _try_gemini(text, tone, length, style)
                if out:
                    print("[DEBUG] ✅ Used Gemini")
                    return out

            elif provider == "offline":
                print("[DEBUG] ⚠️ Falling back to offline summarizer")
                return _offline_summarize(text, tone, length)

        except Exception as e:
            print(f"[ERROR] {provider} failed → {e}")

    # Final fallback (if all fail)
    return _offline_summarize(text, tone, length)

# Back-compat wrapper some code may use:
def smart_summarize(
    text: str,
    provider: str = "openai",
    tone: Tone = "neutral",
    style: str = "simple",
    length: Length = "medium",
    use_ai: bool = True,
) -> str:
    if not use_ai:
        return _offline_summarize(text, tone, length)
    order = ("openai", "gemini", "offline") if provider == "openai" else ("gemini", "openai", "offline")
    return get_summary(text, tone=tone, length=length, style=style, provider_order=order)
