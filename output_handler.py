# output_handler.py
import json
import pandas as pd
from typing import Dict

def save_json(data: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_csv_summary(data: Dict, path: str):
    # convert core fields to a one-row CSV
    row = {
        "title": data.get("meta", {}).get("title", ""),
        "url_method": data.get("method", ""),
        "summary_short": data.get("summary_short", ""),
        "sentiment_label": data.get("sentiment", {}).get("label", ""),
        "sentiment_score": data.get("sentiment", {}).get("score", 0.0),
        "keywords": ",".join(data.get("keywords", [])),
        "entities": ";".join([f"{e['text']}({e['label']})" for e in data.get("entities", [])])
    }
    df = pd.DataFrame([row])
    df.to_csv(path, index=False)

def save_html_report(data: Dict, path: str):
    title = data.get("meta", {}).get("title", "Article")
    html = f"""
    <html><head><meta charset="utf-8"><title>{title}</title></head><body>
    <h1>{title}</h1>
    <h3>Extraction method: {data.get('method')}</h3>
    <h3>Summary (short)</h3><p>{data.get('summary_short')}</p>
    <h3>Summary (tone)</h3><p>{data.get('summary_tone')}</p>
    <h3>Sentiment</h3><p>{data.get('sentiment')}</p>
    <h3>Keywords</h3><p>{', '.join(data.get('keywords', []))}</p>
    <h3>Entities</h3><p>{', '.join([e['text'] + ' (' + e['label'] + ')' for e in data.get('entities', [])])}</p>
    <h3>Reading stats</h3><p>{data.get('stats')}</p>
    <hr>
    <h3>Full text (truncated)</h3><p>{data.get('full_text', '')[:5000]}</p>
    </body></html>
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
