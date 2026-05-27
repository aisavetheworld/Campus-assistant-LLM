"""Automatically discover UCSD source URLs for RAG ingestion.

Strategy per domain:
  1. Try /sitemap.xml first — fast, returns all pages in one shot.
  2. If sitemap unavailable or yields < MIN_SITEMAP_HITS relevant URLs,
     fall back to BFS link crawl from the seed entry URL.

Outputs a candidate URL list to data/rag/discovered_urls.json for human review.
Content is NOT fetched here — review the output, then run fetch_source.py.

Usage:
    python scripts/rag/discover_sources.py
    python scripts/rag/discover_sources.py --output data/rag/discovered_urls.json
    python scripts/rag/discover_sources.py --max_depth 3 --max_per_domain 200
    python scripts/rag/discover_sources.py --domain iseo.ucsd.edu
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("ERROR: pip install requests beautifulsoup4", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADERS = {"User-Agent": "Mozilla/5.0 (research/campus-assistant-llm, non-commercial)"}
REQUEST_DELAY = 1.2       # seconds between requests (be polite)
REQUEST_TIMEOUT = 15
MIN_SITEMAP_HITS = 5      # if sitemap yields fewer, also run BFS

# Patterns that indicate non-content pages — skip these
EXCLUDE_RE = re.compile(
    r"/(login|logout|signin|sso|saml|shibboleth|search|print"
    r"|rss|feed|api/|wp-admin|wp-login|cdn-cgi"
    r"|assets/|static/|img/|images/|fonts/|js/|css/)"
    r"|[?].*print"
    r"|\.pdf$|\.jpg$|\.jpeg$|\.png$|\.gif$|\.zip$|\.doc$|\.docx$",
    re.IGNORECASE,
)

SOURCE_URLS_PATH = Path("data/rag/source_urls.json")
OUTPUT_DEFAULT = Path("data/rag/discovered_urls.json")

# ---------------------------------------------------------------------------
# Domain seed configuration
# Each entry: one domain, one BFS entry URL, one category, keyword list.
# Sitemap is tried automatically for every domain.
# ---------------------------------------------------------------------------

DOMAIN_CONFIGS: list[dict] = [
    # --- International students (ISEO) ---
    {
        "domain": "iseo.ucsd.edu",
        "entry_url": "https://iseo.ucsd.edu/student-services/",
        "category": "international_students",
        "keywords": [
            "visa", "international", "opt", "cpt", "sevis", "i-20", "i20",
            "status", "travel", "authorization", "employment",
            "j-1", "f-1", "arrival", "orientation", "maintain",
            "extension", "reinstate", "reporting", "student-services",
            "curricular", "practical", "depart", "re-entry",
        ],
    },
    # --- General student services portal (housing + enrollment) ---
    {
        "domain": "students.ucsd.edu",
        "entry_url": "https://students.ucsd.edu/",
        "category": "housing",
        "keywords": [
            "mail", "package", "housing", "dorm", "resident", "campus-services",
            "student-mail", "enroll", "add", "drop", "waitlist", "deadline",
            "academic", "grade", "withdraw", "schedule", "quarterly",
        ],
    },
    # --- Registrar / enrollment calendars ---
    {
        "domain": "blink.ucsd.edu",
        "entry_url": "https://blink.ucsd.edu/instructors/courses/enrollment/",
        "category": "course_enrollment",
        # path_prefix: sitemap URLs must start with one of these prefixes
        "path_prefix": ["/instructors/courses", "/students/courses", "/students/academic"],
        "keywords": [
            "enrollment", "waitlist", "calendar",
            "deadline", "grade", "schedule", "withdraw",
        ],
    },
    # --- Undergraduate housing (HDH) ---
    {
        "domain": "hdhughousing.ucsd.edu",
        "entry_url": "https://hdhughousing.ucsd.edu/",
        "category": "housing",
        "keywords": [
            "housing", "room", "assignment", "waitlist", "resident",
            "contract", "faq", "selection", "move", "living",
            "eligibility", "contact", "application", "lottery",
        ],
    },
    # --- Housing facilities / maintenance ---
    {
        "domain": "hdhfacilities.ucsd.edu",
        "entry_url": "https://hdhfacilities.ucsd.edu/",
        "category": "housing",
        "keywords": [
            "maintenance", "fix", "repair", "custodial", "facilities", "fix-it",
        ],
    },
    # --- UC-SHIP health insurance ---
    {
        "domain": "shwadmin.ucsd.edu",
        "entry_url": "https://shwadmin.ucsd.edu/uc-ship/",
        "category": "health_insurance",
        "keywords": [
            "ship", "insurance", "waiver", "coverage", "enroll", "plan",
            "premium", "benefit", "claim", "dependent", "aca", "uc-ship",
            "opt-out", "criteria", "requirement",
        ],
    },
    # --- Student Health Services ---
    {
        "domain": "studenthealth.ucsd.edu",
        "entry_url": "https://studenthealth.ucsd.edu/",
        "category": "student_health",
        "keywords": [
            "appointment", "immunization", "health", "vaccine", "clinic",
            "pharmacy", "counseling", "urgent", "shs", "services",
            "mental", "wellness",
        ],
    },
    # --- Financial Aid & Scholarships (fas.ucsd.edu, not financialaid) ---
    {
        "domain": "fas.ucsd.edu",
        "entry_url": "https://fas.ucsd.edu/",
        "category": "financial_aid",
        "keywords": [
            "types", "aid", "scholarship", "loan", "grant", "cost",
            "fellowship", "award", "stipend", "fafsa", "index",
            "work-study", "tuition", "apply",
        ],
    },
    # --- Graduate Division ---
    {
        "domain": "grad.ucsd.edu",
        "entry_url": "https://grad.ucsd.edu/",
        "category": "international_students",
        # Exclude staff bio pages (/about/meet-the-team/)
        "path_prefix": [
            "/admissions", "/financial", "/funding", "/international",
            "/life", "/academic", "/degrees", "/current", "/resources",
            "/registration", "/health",
        ],
        "keywords": [
            "international", "visa", "funding", "housing", "health",
            "registration", "fee", "stipend", "diversity", "academic",
            "financial", "admission", "degree", "current",
        ],
    },
]

# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    p = urlparse(url)
    # Strip fragment and query string, remove trailing slash
    clean = p._replace(fragment="", query="").geturl()
    return clean.rstrip("/")


def is_excluded(url: str) -> bool:
    return bool(EXCLUDE_RE.search(url))


def url_matches(url: str, keywords: list[str]) -> bool:
    lower = url.lower()
    return any(kw in lower for kw in keywords)


def text_matches(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def safe_get(url: str, session: requests.Session) -> requests.Response | None:
    try:
        resp = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        return resp
    except requests.exceptions.HTTPError as e:
        print(f"    HTTP {e.response.status_code}: {url}")
        return None
    except Exception as e:
        print(f"    WARN: {url} — {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Sitemap discovery
# ---------------------------------------------------------------------------

def _parse_sitemap_xml(content: str, domain: str, keywords: list[str],
                       path_prefix: list[str]) -> tuple[list[str], list[str]]:
    """Return (matched_page_urls, child_sitemap_urls)."""
    soup = BeautifulSoup(content, "xml")

    # Sitemap index: <sitemapindex> contains <sitemap><loc>
    child_sitemaps: list[str] = []
    for tag in soup.find_all("sitemap"):
        loc = tag.find("loc")
        if loc and domain in loc.get_text():
            child_sitemaps.append(loc.get_text(strip=True))

    if child_sitemaps:
        return [], child_sitemaps

    # Regular sitemap: <urlset> contains <url><loc>
    pages: list[str] = []
    for loc_tag in soup.find_all("loc"):
        url = normalize_url(loc_tag.get_text(strip=True))
        if domain not in urlparse(url).netloc:
            continue
        if is_excluded(url):
            continue
        # Path prefix filter: if configured, URL must start with one of the prefixes
        if path_prefix:
            path = urlparse(url).path
            if not any(path.startswith(p) for p in path_prefix):
                continue
        if url_matches(url, keywords):
            pages.append(url)

    return pages, []


def try_sitemap(cfg: dict, session: requests.Session) -> list[dict]:
    domain = cfg["domain"]
    sitemap_url = f"https://{domain}/sitemap.xml"
    print(f"  Trying sitemap: {sitemap_url}")

    resp = safe_get(sitemap_url, session)
    time.sleep(REQUEST_DELAY)

    if resp is None:
        print("  No sitemap.")
        return []
    content_type = resp.headers.get("Content-Type", "")
    if "xml" not in content_type and "text" not in content_type:
        print(f"  Sitemap returned unexpected Content-Type: {content_type}")
        return []

    pending_sitemaps = [resp.text]
    visited_sitemaps: set[str] = {sitemap_url}
    all_pages: list[str] = []
    path_prefix: list[str] = cfg.get("path_prefix", [])

    while pending_sitemaps:
        content = pending_sitemaps.pop(0)
        pages, children = _parse_sitemap_xml(content, domain, cfg["keywords"], path_prefix)
        all_pages.extend(pages)
        for child_url in children:
            if child_url not in visited_sitemaps:
                visited_sitemaps.add(child_url)
                child_resp = safe_get(child_url, session)
                time.sleep(REQUEST_DELAY)
                if child_resp:
                    pending_sitemaps.append(child_resp.text)

    # Deduplicate
    seen: set[str] = set()
    results: list[dict] = []
    for url in all_pages:
        if url not in seen:
            seen.add(url)
            results.append({
                "url": url,
                "domain": domain,
                "category": cfg["category"],
                "discovery_method": "sitemap",
                "depth": 0,
            })

    print(f"  Sitemap: {len(results)} relevant URLs")
    return results


# ---------------------------------------------------------------------------
# BFS link crawl
# ---------------------------------------------------------------------------

def bfs_crawl(cfg: dict, max_depth: int, max_pages: int,
              session: requests.Session) -> list[dict]:
    domain = cfg["domain"]
    entry = normalize_url(cfg["entry_url"])
    print(f"  BFS crawl from {entry} (depth≤{max_depth}, max {max_pages} pages)")

    queue: deque[tuple[str, int]] = deque([(entry, 0)])
    visited: set[str] = {entry}
    discovered: list[dict] = []

    while queue and len(discovered) < max_pages:
        url, depth = queue.popleft()

        resp = safe_get(url, session)
        time.sleep(REQUEST_DELAY)
        if resp is None:
            continue

        discovered.append({
            "url": url,
            "domain": domain,
            "category": cfg["category"],
            "discovery_method": "bfs",
            "depth": depth,
        })

        if depth >= max_depth:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag.get("href", "").strip()
            if not href or href.startswith(("mailto:", "tel:", "javascript:")):
                continue

            abs_url = normalize_url(urljoin(url, href))
            parsed = urlparse(abs_url)

            if parsed.netloc != domain:
                continue
            if is_excluded(abs_url):
                continue
            if abs_url in visited:
                continue

            # Only follow links that seem relevant
            link_text = a_tag.get_text(strip=True)
            if url_matches(abs_url, cfg["keywords"]) or text_matches(link_text, cfg["keywords"]):
                visited.add(abs_url)
                queue.append((abs_url, depth + 1))

    print(f"  BFS: {len(discovered)} pages crawled")
    return discovered


# ---------------------------------------------------------------------------
# Per-domain discovery
# ---------------------------------------------------------------------------

def discover_domain(cfg: dict, args: argparse.Namespace,
                    session: requests.Session) -> list[dict]:
    domain = cfg["domain"]
    print(f"\n[{domain}]  category={cfg['category']}")

    results = try_sitemap(cfg, session)

    if len(results) < MIN_SITEMAP_HITS:
        bfs_results = bfs_crawl(cfg, args.max_depth, args.max_per_domain, session)
        existing = {r["url"] for r in results}
        for r in bfs_results:
            if r["url"] not in existing:
                results.append(r)
                existing.add(r["url"])

    return results


# ---------------------------------------------------------------------------
# Load known URLs from existing source_urls.json
# ---------------------------------------------------------------------------

def load_known_urls() -> set[str]:
    if not SOURCE_URLS_PATH.exists():
        return set()
    data = json.loads(SOURCE_URLS_PATH.read_text())
    if isinstance(data, dict):
        return {normalize_url(v) for v in data.values() if v}
    return {normalize_url(e.get("url", "")) for e in data if e.get("url")}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover UCSD source URLs for RAG.")
    parser.add_argument("--output", default=str(OUTPUT_DEFAULT),
                        help="Output JSON path (default: data/rag/discovered_urls.json)")
    parser.add_argument("--max_depth", type=int, default=3,
                        help="BFS max link depth per domain (default: 3)")
    parser.add_argument("--max_per_domain", type=int, default=150,
                        help="BFS max pages per domain (default: 150)")
    parser.add_argument("--domain", default="",
                        help="Process only this domain (e.g. iseo.ucsd.edu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    known_urls = load_known_urls()
    print(f"Already in source_urls.json: {len(known_urls)} known URLs")

    configs = DOMAIN_CONFIGS
    if args.domain:
        configs = [c for c in DOMAIN_CONFIGS if c["domain"] == args.domain]
        if not configs:
            available = ", ".join(c["domain"] for c in DOMAIN_CONFIGS)
            print(f"ERROR: domain '{args.domain}' not in config.\nAvailable: {available}",
                  file=sys.stderr)
            sys.exit(1)

    all_results: list[dict] = []
    seen_urls: set[str] = set()

    with requests.Session() as session:
        for cfg in configs:
            results = discover_domain(cfg, args, session)
            for r in results:
                if r["url"] not in seen_urls:
                    r["already_known"] = r["url"] in known_urls
                    all_results.append(r)
                    seen_urls.add(r["url"])

    # New URLs first, then by domain + url
    all_results.sort(key=lambda r: (r["already_known"], r["domain"], r["url"]))

    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False) + "\n")

    # Summary
    new_count = sum(1 for r in all_results if not r["already_known"])
    known_count = sum(1 for r in all_results if r["already_known"])
    by_domain: dict[str, int] = {}
    for r in all_results:
        by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + 1

    print(f"\n{'='*60}")
    print(f"Total discovered:  {len(all_results)}")
    print(f"  New URLs:        {new_count}")
    print(f"  Already known:   {known_count}")
    print(f"\nPer domain:")
    for domain, count in sorted(by_domain.items()):
        print(f"  {domain}: {count}")
    print(f"\nSaved → {out_path}")
    print(f"\nNext steps:")
    print(f"  1. Review {out_path} — remove any irrelevant URLs")
    print(f"  2. Add chosen URLs to data/rag/source_urls.json")
    print(f"  3. Run: python scripts/rag/fetch_source.py --batch '{{...}}'")


if __name__ == "__main__":
    main()
