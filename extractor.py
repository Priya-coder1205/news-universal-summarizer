# extractor.py
from typing import Dict
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Setup a session with retries + headers
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
})

def extract_article(url: str) -> Dict[str, str]:
    data = {"title": "", "text": "", "url": url}

    # --- Try Newspaper3k ---
    try:
        article = Article(url)
        article.download()
        article.parse()
        if article.text.strip():
            data["title"] = article.title or ""
            data["text"] = article.text.strip()
            return data
    except Exception as e:
        print(f"[Extractor] Newspaper3k failed: {e}")

    # --- Fallback: BeautifulSoup ---
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")

            # Title
            if not data["title"]:
                title_tag = soup.find("title")
                if title_tag:
                    data["title"] = title_tag.get_text().strip()

            # Text from <p>
            paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
            text = " ".join(paragraphs)
            if text:
                data["text"] = text
    except Exception as e:
        print(f"[Extractor] BeautifulSoup fallback failed: {e}")

    return data


# --- Debug usage ---
if __name__ == "__main__":
    test_url = "https://www.thehindu.com/news/cities/mumbai/two-dead-over-300-injured-in-dahi-handi-festivities-in-mumbai/article69944704.ece"
    result = extract_article(test_url)
    print("Title:", result.get("title", "(no title)"))
    print("Text:", result.get("text", "(no text)")[:500], "...")
