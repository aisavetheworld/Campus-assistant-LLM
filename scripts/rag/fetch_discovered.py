"""Batch-fetch text for all new URLs in data/rag/discovered_urls.json.

Assigns source_ids automatically based on domain, saves raw text to
data/rag/raw_sources/, and writes a fetch manifest for later use.

Resumable: already-fetched files are skipped. Re-run anytime to pick up
where you left off after failures or interruptions.

Usage:
    python scripts/rag/fetch_discovered.py
    python scripts/rag/fetch_discovered.py --delay 1.5
    python scripts/rag/fetch_discovered.py --domain iseo.ucsd.edu
    python scripts/rag/fetch_discovered.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("ERROR: pip install requests beautifulsoup4", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DISCOVERED_PATH = Path("data/rag/discovered_urls.json")
RAW_DIR = Path("data/rag/raw_sources")
MANIFEST_PATH = Path("data/rag/fetch_manifest.json")

# ---------------------------------------------------------------------------
# source_id prefix per domain
# ---------------------------------------------------------------------------

DOMAIN_PREFIX = {
    "iseo.ucsd.edu":          "ucsd_iseo",
    "students.ucsd.edu":      "ucsd_students",
    "blink.ucsd.edu":         "ucsd_blink",
    "hdhughousing.ucsd.edu":  "ucsd_hdh_ug",
    "hdhfacilities.ucsd.edu": "ucsd_hdh_fac",
    "shwadmin.ucsd.edu":      "ucsd_ucship",
    "studenthealth.ucsd.edu": "ucsd_shs",
    "fas.ucsd.edu":           "ucsd_fas",
    "grad.ucsd.edu":          "ucsd_grad",
}

# ---------------------------------------------------------------------------
# HTTP helpers (same logic as fetch_source.py)
# ---------------------------------------------------------------------------

HEADERS = {"User-Agent": "Mozilla/5.0 (research/campus-assistant-llm, non-commercial)"}
TIMEOUT = 15

REMOVE_TAGS = ["nav", "header", "footer", "script", "style",
               "noscript", "aside", "form", "iframe"]

CONTENT_SELECTORS = [
    "main", "article", '[role="main"]',
    "#main-content", "#content", ".content",
    ".page-content", ".entry-content",
]


def fetch_text(url: str, session: requests.Session) -> str:
    resp = session.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(REMOVE_TAGS):
        tag.decompose()

    body = None
    for sel in CONTENT_SELECTORS:
        body = soup.select_one(sel)
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

# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

# ---------------------------------------------------------------------------
# source_id assignment
# ---------------------------------------------------------------------------

def assign_source_ids(entries: list[dict], manifest: dict) -> list[tuple[str, dict]]:
    """Pair each entry with a source_id. Re-uses IDs from manifest if URL seen before."""
    url_to_sid = {v["url"]: k for k, v in manifest.items()}

    # Track next counter per domain prefix
    prefix_counters: dict[str, int] = {}
    for sid in manifest:
        for prefix in DOMAIN_PREFIX.values():
            if sid.startswith(prefix + "_"):
                suffix = sid[len(prefix) + 1:]
                if suffix.isdigit():
                    prefix_counters[prefix] = max(
                        prefix_counters.get(prefix, 0), int(suffix)
                    )

    pairs: list[tuple[str, dict]] = []
    for entry in entries:
        url = entry["url"]
        if url in url_to_sid:
            pairs.append((url_to_sid[url], entry))
            continue

        domain = entry["domain"]
        prefix = DOMAIN_PREFIX.get(domain, "ucsd_other")
        n = prefix_counters.get(prefix, 0) + 1
        prefix_counters[prefix] = n
        sid = f"{prefix}_{n:04d}"
        pairs.append((sid, entry))

    return pairs

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch fetch discovered UCSD URLs.")
    parser.add_argument("--delay", type=float, default=1.2,
                        help="Seconds between requests (default: 1.2)")
    parser.add_argument("--domain", default="",
                        help="Only fetch URLs for this domain")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be fetched, don't fetch")
    parser.add_argument("--min_words", type=int, default=30,
                        help="Skip saving if fetched text has fewer words (default: 30)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not DISCOVERED_PATH.exists():
        print(f"ERROR: {DISCOVERED_PATH} not found. Run discover_sources.py first.")
        sys.exit(1)

    discovered: list[dict] = json.loads(DISCOVERED_PATH.read_text())
    manifest: dict = load_manifest()

    # Filter to new URLs only
    to_fetch = [r for r in discovered if not r["already_known"]]
    if args.domain:
        to_fetch = [r for r in to_fetch if r["domain"] == args.domain]

    pairs = assign_source_ids(to_fetch, manifest)

    # Split: already done vs pending
    pending = [(sid, entry) for sid, entry in pairs
               if not (RAW_DIR / f"{sid}.txt").exists()]
    done_count = len(pairs) - len(pending)

    print(f"Total new URLs:    {len(pairs)}")
    print(f"Already fetched:   {done_count}")
    print(f"To fetch now:      {len(pending)}")
    if args.dry_run:
        print("\n[dry_run] Would fetch:")
        for sid, entry in pending[:20]:
            print(f"  {sid}  {entry['url']}")
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")
        return

    if not pending:
        print("Nothing to fetch.")
        return

    eta_min = len(pending) * args.delay / 60
    print(f"Estimated time:    {eta_min:.1f} min at {args.delay}s/request\n")

    stats = {"ok": 0, "empty": 0, "error": 0}
    today = date.today().isoformat()

    with requests.Session() as session:
        for i, (sid, entry) in enumerate(pending, 1):
            url = entry["url"]
            out_path = RAW_DIR / f"{sid}.txt"

            print(f"[{i}/{len(pending)}] {sid}")
            print(f"  {url}")

            try:
                text = fetch_text(url, session)
            except requests.exceptions.HTTPError as e:
                print(f"  HTTP {e.response.status_code} — skip")
                stats["error"] += 1
                time.sleep(args.delay)
                continue
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e} — skip")
                stats["error"] += 1
                time.sleep(args.delay)
                continue

            word_count = len(text.split())
            if word_count < args.min_words:
                print(f"  too short ({word_count} words) — skip")
                stats["empty"] += 1
                time.sleep(args.delay)
                continue

            out_path.write_text(text, encoding="utf-8")
            print(f"  saved ({word_count} words)")
            stats["ok"] += 1

            # Update manifest immediately so partial runs are recoverable
            manifest[sid] = {
                "url": url,
                "domain": entry["domain"],
                "category": entry["category"],
                "date_fetched": today,
                "word_count": word_count,
            }
            save_manifest(manifest)

            time.sleep(args.delay)

    # Final summary
    print(f"\n{'='*60}")
    print(f"Fetched:  {stats['ok']} files saved")
    print(f"Skipped (too short / empty): {stats['empty']}")
    print(f"Errors:   {stats['error']}")
    print(f"Manifest: {MANIFEST_PATH}  ({len(manifest)} total entries)")
    print(f"\nNext: review {MANIFEST_PATH}, then run chunk_sources.py")


if __name__ == "__main__":
    main()
