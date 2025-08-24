# app_streamlit.py
import os
from dotenv import load_dotenv

#print("[DEBUG] Loading .env file...")
load_dotenv()

#print("[DEBUG] OPENAI_API_KEY after load:", os.getenv("OPENAI_API_KEY"))
#print("[DEBUG] GEMINI_API_KEY after load:", os.getenv("GEMINI_API_KEY"))

import json
import io
import textwrap
from typing import List
import streamlit as st


import extractor
from textblob import TextBlob
from summarizer_api import get_summary  # keep this for summarization
import bias_checker  # our upgraded bias module (compare + visualize)

# ---------- Page & Theming ----------
st.set_page_config(page_title="Universal News Summarizer", page_icon="📰", layout="wide")

# default theme once per session
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

# inject both themes; swap via [data-theme]
st.markdown("""
<style>
/* 🌞 Light theme */
[data-theme="light"] .stApp { background: #f7fafc; color: #111827; }
[data-theme="light"] .card { background: #ffffff; border: 1px solid #e5e7eb; box-shadow: 0 10px 22px rgba(0,0,0,.06); }
[data-theme="light"] h1, [data-theme="light"] h2, [data-theme="light"] h3,
[data-theme="light"] .stMarkdown { color: #0f172a !important; }
[data-theme="light"] .pill { background: #eef2ff; color: #3730a3; border-color: #c7d2fe; }
[data-theme="light"] .stTextInput>div>div>input, 
[data-theme="light"] .stTextArea textarea {
  background: #ffffff !important; color:#111827 !important; border-color:#cbd5e1 !important;
}
[data-theme="light"] .stButton>button { 
  background: linear-gradient(180deg, #ffffff 0%, #f3f4f6 100%) !important; 
  color:#111827 !important; 
}

/* 🌙 Dark theme */
[data-theme="dark"] .stApp { background: #0f172a; color: #f1f5f9; }
[data-theme="dark"] .card { background: #1e293b; border: 1px solid #334155; box-shadow: 0 10px 22px rgba(0,0,0,.4); }
[data-theme="dark"] h1, [data-theme="dark"] h2, [data-theme="dark"] h3,
[data-theme="dark"] .stMarkdown { color: #f1f5f9 !important; }
[data-theme="dark"] .pill { background: #334155; color: #e2e8f0; border-color: #475569; }
[data-theme="dark"] .stTextInput>div>div>input, 
[data-theme="dark"] .stTextArea textarea {
  background: #1e293b !important; color:#f1f5f9 !important; border-color:#475569 !important;
}
[data-theme="dark"] .stButton>button { 
  background: linear-gradient(180deg, #334155 0%, #1e293b 100%) !important; 
  color:#f1f5f9 !important; 
}
</style>
""", unsafe_allow_html=True)

# set initial theme attribute on first render
st.markdown(f"""
<script>
window.addEventListener('load', () => {{
  const root = window.parent.document.documentElement;
  root.setAttribute('data-theme', '{st.session_state["theme"]}');
}});
</script>
""", unsafe_allow_html=True)

# Sidebar: Info Button
with st.sidebar:
    # Info toggle button
    if "show_info" not in st.session_state:
        st.session_state.show_info = False

    if st.button("ℹ️ Info", use_container_width=True):
        st.session_state.show_info = not st.session_state.show_info

    # Show About panel only when button is active
    if st.session_state.show_info:
        st.markdown(
            """
            ## 📖 About  
            **Universal News Summarizer**  
            Extracts & summarizes news from multiple sources.  

            ### ✨ Features
            - 🔗 Multiple URL support  
            - 📄 PDF / TXT file input  
            - 🧠 AI-powered summarization (with offline fallback)  
            - 📝 Keyword extraction  
            - 📊 Sentiment & bias detection  

            ---

            ## 🛠️ How to Use
            1. Paste one or more URLs in the input box  
            2. Or upload a **PDF/TXT file**  
            3. Choose summarization length & options  
            4. Click **Extract & Summarize**  
            5. View **Summary, Keywords & Sentiment**  

            ✅ That’s it — your personalized news digest!
            """
        )
        st.caption("Built with ❤️ using Streamlit + NLP")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def _extract_single(url: str) -> dict:
    return extractor.extract_article(url)


def _extract_many(urls: List[str]) -> List[dict]:
    out = []
    for u in urls:
        u = u.strip()
        if not u:
            continue
        try:
            out.append(_extract_single(u))
        except Exception as e:
            out.append({"status": "fail", "method": "none", "title": "", "description": "", "text": "", "url": u, "error": str(e)})
    return out

def _sentiment_score(text: str) -> str:
    try:
        s = TextBlob(text).sentiment.polarity
        if s > 0.15: return f"Positive ({s:.2f})"
        if s < -0.15: return f"Negative ({s:.2f})"
        return f"Neutral ({s:.2f})"
    except Exception:
        return "N/A"

def _keywords(text: str, k: int = 8) -> List[str]:
    import re, collections
    words = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", text.lower())
    stop = set("""
a about above after again against all am an and any are aren't as at be because been before being below between
both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from
further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his
how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor
not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll
they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what
what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())
    words = [w for w in words if w not in stop]
    counts = collections.Counter(words).most_common(k)
    return [w for w,c in counts]

def _merge_texts(items: List[dict]) -> str:
    parts = []
    for it in items:
        if it.get("text"):
            parts.append(it["text"])
    return "\n\n".join(parts)

def _download_bytes(name: str, text: str) -> bytes:
    return text.encode("utf-8")

# ---------- UI ----------
st.title("📰 Universal News Summarizer")

st.write("Paste one or multiple URLs (one per line), or upload files (PDF/TXT). I’ll extract, analyze, and summarize.")

colA, colB = st.columns([2,1])

with colA:
    urls_raw = st.text_area("URLs (one per line)", height=120, placeholder="https://example.com/article-1\nhttps://example.com/article-2")
with colB:
    st.markdown("**Upload Files**")
    files = st.file_uploader("PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True, label_visibility="collapsed")

st.markdown("### Summary Controls")
c1, c2 = st.columns([1,1])

with c1:
    length = st.selectbox("Length", ["short", "medium", "long"], index=1)
with c2:
    use_ai = st.toggle("Use AI (if keys set)", value=True, help="Uses OpenAI/Gemini if API keys are provided; otherwise offline.")


# --- Tone options ---
tone_choice = st.selectbox(
    "Choose summarization tone:",
    ["Neutral summary", "Fact-only", "Explain to a 10-year-old"]
)

# Map UI choices to internal codes your summarizer expects
tone_map = {
    "Neutral summary": "neutral",
    "Fact-only": "fact-only",
    "Explain to a 10-year-old": "kid-friendly"  # <- important
}


tone = tone_map.get(tone_choice, "neutral")


pref = st.selectbox("Preferred Engine", ["auto", "openai", "gemini"], index=0, help="Order to try when AI is enabled.")

go = st.button("Extract & Summarize", type="primary")

# ---------- Run ----------
if go:
    collected = []

    # 1) URLs
    url_list = [u.strip() for u in urls_raw.splitlines() if u.strip()]
    if url_list:
        with st.spinner("Extracting from URLs..."):
            results = _extract_many(url_list)
            for r in results:
                collected.append(r)

    # 2) Files (PDF/TXT)
    if files:
        import tempfile
        from pdfminer.high_level import extract_text as pdf_text
        for f in files:
            try:
                if f.type == "text/plain":
                    txt = f.read().decode("utf-8", errors="ignore")
                    collected.append({"status": "ok", "method": "upload", "title": f.name, "description": "", "text": txt, "url": f.name})
                elif f.type == "application/pdf":
                    # Save temp and parse
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.read())
                        tmp.flush()
                        try:
                            t = pdf_text(tmp.name) or ""
                        except Exception:
                            t = ""
                    collected.append({"status": "ok", "method": "pdf-upload", "title": f.name, "description": "", "text": t, "url": f.name})
            except Exception as e:
                collected.append({"status": "fail", "method": "upload", "title": f.name, "description": "", "text": "", "url": f.name, "error": str(e)})

    if not collected:
        st.warning("Provide at least one URL or upload a file.")
        st.stop()

    # Show per-source cards
    st.markdown("## Sources")
    for idx, item in enumerate(collected, 1):
        with st.container():
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            cols = st.columns([3,1])
            with cols[0]:
                st.markdown(f"**[{idx}] {item.get('title','(No title)')}**")
                st.caption(item.get("url","N/A"))
                st.markdown(f'<span class="pill">{item.get("status","")}</span> <span class="pill">{item.get("method","")}</span>', unsafe_allow_html=True)
            with cols[1]:
                if item.get("text"):
                    st.caption("Sentiment")
                    st.code(_sentiment_score(item["text"]), language="text")
            with st.expander("Show extracted", expanded=False):
                if "text" in item and item["text"].strip():
                    st.write(item["text"])
                else:
                    st.warning("No text was extracted from this URL.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("")

    # Merge texts for global summary
    merged = _merge_texts(collected).strip()
    if not merged:
        st.error("Could not extract any text to summarize.")
        st.stop()

    # Keywords (quick)
    st.markdown("## 🔑 Keywords")
    kws = _keywords(merged, k=10)
    st.write(", ".join(kws) if kws else "N/A")

    # --- Bias & Sentiment (across sources) ---
    st.markdown("## 🕵️ Bias & Sentiment Insights")

    bias_report = bias_checker.compare(collected)

    # Show overall bias message (consistent vs diverging)
    if bias_report.get("flag", False):
        st.warning(f"⚠️ {bias_report['message']}")
    else:
        st.success(f"✅ {bias_report['message']}")

    # Deduplicate sources by URL or title
    unique_sources = {}
    for item in bias_report.get("details", []):
        key = item.get("url") or item.get("title")
        if key and key not in unique_sources:
            unique_sources[key] = item
    sources_to_show = list(unique_sources.values())

    # Colors for sentiment labels (bright and visible)
    sentiment_colors = {
        "Positive": {"bg": "#16a34a", "fg": "white"},  # green
        "Neutral": {"bg": "#2563eb", "fg": "white"},   # blue
        "Negative": {"bg": "#dc2626", "fg": "white"}   # red
    }

    # Display each source nicely
    for src in sources_to_show:
        cols = st.columns([3, 1, 2])
        with cols[0]:
            st.markdown(f"**{src.get('title','(No title)')}**")
            st.caption(src.get("url",""))
        with cols[1]:
            label = src.get("label", "Unknown")
            color = sentiment_colors.get(label, {"bg": "#6b7280", "fg": "white"})  # fallback gray
            st.markdown(
                f'<div style="background:{color["bg"]};color:{color["fg"]};'
                f'padding:6px;border-radius:8px;text-align:center;font-weight:600">'
                f'{label}</div>', unsafe_allow_html=True
            )
        with cols[2]:
            flagged = ", ".join(src.get("flagged_words", [])) or "None"
            st.caption(f"**Flagged words:** {flagged}")

    # Simple bias check message instead of numeric spread
    if bias_report.get("spread", 0.0) > 0.3:  # tweak threshold
        st.warning("⚠️ Sentiment diverges across sources (possible bias).")
    else:
        st.success("✅ Sentiment is consistent across sources.")

    
    # ---------- Summarize ----------
    with st.spinner("Summarizing..."):
        # Provider order logic
        if use_ai:
            if pref == "openai":
                order = ("openai", "gemini", "offline")
            elif pref == "gemini":
                order = ("gemini", "openai", "offline")
            else:
                order = ("openai", "gemini", "offline")
        else:
            order = ("offline",)
        
        

        # Call summarizer on the MERGED text
        summary_text = get_summary(
            merged,
            tone=tone,
            length=length,
            style="concise",           
            provider_order=order
        )

        engine_label = "Offline" if order == ("offline",) else f"{order[0].capitalize()} → fallback"
        sres = {"engine": engine_label, "summary": summary_text}

        st.markdown("## 📝 Summary")
        if tone == "fact-only":
            st.markdown("### Fact Highlights")
            for line in summary_text.split(". "):
                if line.strip():
                    st.markdown(f"- {line.strip()}")
        else:
            st.write(summary_text)

        # Copy & Download
        d1_col1, d1_col2 = st.columns([1,1])

        with d1_col1:
            st.download_button(
                label="Download summary (.txt)",
                data=summary_text.encode("utf-8"),
                file_name="summary.txt",
                mime="text/plain",
            )

        with d1_col2:
            # Export JSON (sources + summary)
            package = {
                "engine": sres['engine'],
                "tone": tone,
                "style": "concise",  # ✅ save style string separately
                "length": length,
                "summary": summary_text,
                "sources": collected,
                "keywords": kws
            }

            st.download_button(
                label="Download package (.json)",
                data=json.dumps(package, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="summary_package.json",
                mime="application/json"
            )
else:
    st.info("Enter URLs and/or upload files, then click \"Extract & Summarize\".")

