"""Fetch an official source page and save cleaned body text.

Usage:
    python scripts/rag/fetch_source.py \
        --source_id ucsd_iseo_visa_status_001 \
        --url "https://ispo.ucsd.edu/advising/visa/..."

    # Fetch multiple at once using a JSON mapping:
    python scripts/rag/fetch_source.py \
        --batch '{"ucsd_iseo_visa_status_001": "https://...", "ucsd_student_mail_001": "https://..."}'

Saves to:
    data/rag/raw_sources/<source_id>.txt
    Also updates url and date_collected in data/rag/ucsd_sources.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("ERROR: missing dependencies. Run:\n  pip install requests beautifulsoup4", file=sys.stderr)
    sys.exit(1)

SOURCES_META = Path("data/rag/ucsd_sources.json")
RAW_DIR = Path("data/rag/raw_sources")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research project, campus assistant LLM)"
}

REMOVE_TAGS = ["nav", "header", "footer", "script", "style", "noscript", "aside", "form"]

CONTENT_SELECTORS = [
    "main",
    "article",
    '[role="main"]',
    "#main-content",
    "#content",
    ".content",
    ".page-content",
    ".entry-content",
]


def fetch_text(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(REMOVE_TAGS):
        tag.decompose()

    # Try content selectors first
    body = None
    for selector in CONTENT_SELECTORS:
        body = soup.select_one(selector)
        if body:
            break

    if body is None:
        body = soup.find("body") or soup

    lines = []
    for elem in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
        text = elem.get_text(separator=" ", strip=True)
        if text and len(text) > 15:
            lines.append(text)

    return "\n\n".join(lines)


def update_sources_meta(source_id: str, url: str) -> None:
    if not SOURCES_META.exists():
        return
    sources = json.loads(SOURCES_META.read_text())
    today = date.today().isoformat()
    for entry in sources:
        if entry["source_id"] == source_id:
            entry["url"] = url
            entry["date_collected"] = today
            break
    SOURCES_META.write_text(json.dumps(sources, indent=2, ensure_ascii=False) + "\n")


def fetch_one(source_id: str, url: str) -> None:
    print(f"  Fetching {source_id} from {url}")
    try:
        text = fetch_text(url)
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    out_path = RAW_DIR / f"{source_id}.txt"
    out_path.write_text(text, encoding="utf-8")
    word_count = len(text.split())
    print(f"  Saved {out_path} ({word_count} words)")

    update_sources_meta(source_id, url)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and save official source text.")
    parser.add_argument("--source_id", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--batch", default="",
                        help='JSON string: {"source_id": "url", ...}')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.batch:
        pairs = json.loads(args.batch)
        for source_id, url in pairs.items():
            fetch_one(source_id, url)
            time.sleep(1)
    elif args.source_id and args.url:
        fetch_one(args.source_id, args.url)
    else:
        print("ERROR: provide --source_id + --url, or --batch", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
