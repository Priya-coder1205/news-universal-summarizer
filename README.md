# Universal Article Extractor (Phase 1)

Layered extractor that works on most public URLs:
1) newspaper3k → 2) BeautifulSoup → 3) UA rotation → 4) Playwright (optional) → 5) meta fallback.
Also supports **PDF** (pdfminer).

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -c "import nltk; import ssl; ssl._create_default_https_context = ssl._create_unverified_context; nltk.download('punkt')"