# test_extractor.py
import json
import sys
from extractor import extract

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_extractor.py <url>")
        sys.exit(1)

    url = sys.argv[1]
    out = extract(url)

    print("\n=== EXTRACTION RESULT ===")
    print(f"Status : {out['status']}")
    print(f"Method : {out['method']}")
    meta = out.get("meta", {})
    print(f"Title  : {meta.get('title','')}")
    print(f"Desc   : {meta.get('description','')}")
    print(f"Chars  : {len(out.get('text',''))}")
    print("\n--- Preview ---")
    print(out.get("text","")[:800], "..." if len(out.get("text",""))>800 else "")
    print("\n--- JSON (copyable) ---")
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
