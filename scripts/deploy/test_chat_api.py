"""
Manual smoke test for the Campus Assistant Chat API.

Usage:
    python scripts/deploy/test_chat_api.py
    python scripts/deploy/test_chat_api.py --top-k 3
    python scripts/deploy/test_chat_api.py --url http://localhost:8080
    CHAT_API_URL=http://localhost:8080 python scripts/deploy/test_chat_api.py
"""

import argparse
import os
import sys

import httpx

API_URL = os.getenv("CHAT_API_URL", "http://localhost:8080")

TEST_QUERIES = [
    {
        "query": "Can I start CPT before my advisor approves it?",
        "category": "CPT/OPT",
    },
    {
        "query": "How do I waive UC SHIP health insurance?",
        "category": "health insurance",
    },
    {
        "query": "My package was not delivered to the mailroom. What should I do?",
        "category": "housing/mailroom",
    },
    {
        "query": "I have an immunization hold preventing registration. How do I clear it?",
        "category": "immunization",
    },
    {
        "query": "What happens to my GPA if I drop a class after week 2?",
        "category": "academic/W grade",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test the Campus Assistant Chat API with representative queries."
    )
    parser.add_argument(
        "--url",
        default=API_URL,
        help="Base URL of the chat API (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of sources to retrieve for each query (default: %(default)s)",
    )
    return parser.parse_args()


def print_separator(char: str = "=", width: int = 60) -> None:
    print(char * width)


def print_result(index: int, item: dict, response: dict) -> None:
    print_separator()
    print(f"Query {index}: [{item['category']}]")
    print(f"  {item['query']}")
    print()

    print("Answer:")
    answer = response.get("answer", "(no answer)")
    for line in answer.splitlines():
        print(f"  {line}")
    print()

    sources = response.get("sources", [])
    print(f"Sources ({len(sources)}):")
    if sources:
        for src in sources:
            rank = src.get("rank", "?")
            title = src.get("title", "(no title)")
            url = src.get("url", "(no url)")
            print(f"  [{rank}] {title}")
            print(f"       {url}")
    else:
        print("  (none)")
    print()

    safety = response.get("safety_metadata", {})
    all_passed = safety.get("all_passed", None)
    failed_checks = safety.get("failed_checks", [])
    print(f"Safety: all_passed={all_passed}", end="")
    if failed_checks:
        print(f"  failed_checks={failed_checks}", end="")
    print()
    print()

    latency = response.get("latency", {})
    retrieval_ms = latency.get("retrieval_ms", "N/A")
    generation_ms = latency.get("generation_ms", "N/A")
    total_ms = latency.get("total_ms", "N/A")
    print(
        f"Latency: retrieval={retrieval_ms}ms  generation={generation_ms}ms  total={total_ms}ms"
    )


def main() -> None:
    args = parse_args()
    base_url = args.url.rstrip("/")
    top_k = args.top_k
    endpoint = f"{base_url}/chat"

    print_separator("=")
    print("Campus Assistant Chat API — Smoke Test")
    print(f"Endpoint : {endpoint}")
    print(f"top_k    : {top_k}")
    print(f"Queries  : {len(TEST_QUERIES)}")
    print_separator("=")

    total_queries = len(TEST_QUERIES)
    passed_safety = 0
    failed_safety = 0
    total_latencies: list[float] = []

    try:
        with httpx.Client(timeout=60.0) as client:
            for i, item in enumerate(TEST_QUERIES, start=1):
                payload = {"query": item["query"], "top_k": top_k}
                try:
                    resp = client.post(endpoint, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.HTTPStatusError as exc:
                    print_separator()
                    print(f"Query {i}: [{item['category']}]")
                    print(f"  {item['query']}")
                    print(f"  ERROR: HTTP {exc.response.status_code} — {exc.response.text}")
                    failed_safety += 1
                    continue

                print_result(i, item, data)

                safety = data.get("safety_metadata", {})
                if safety.get("all_passed", False):
                    passed_safety += 1
                else:
                    failed_safety += 1

                latency = data.get("latency", {})
                total_ms = latency.get("total_ms")
                if total_ms is not None:
                    total_latencies.append(float(total_ms))

    except httpx.ConnectError:
        print()
        print("ERROR: Could not connect to the chat API.")
        print(f"  Make sure the server is running at: {base_url}")
        print("  You can override the URL with --url or CHAT_API_URL env var.")
        sys.exit(1)
    except httpx.TimeoutException:
        print()
        print("ERROR: Request timed out (>60s). The server may be overloaded.")
        sys.exit(1)

    # Summary
    print_separator("=")
    print("Summary")
    print_separator("-", 40)
    print(f"  Total queries      : {total_queries}")
    print(f"  Safety passed      : {passed_safety}")
    print(f"  Safety failed      : {failed_safety}")
    if total_latencies:
        avg_ms = sum(total_latencies) / len(total_latencies)
        print(f"  Avg total latency  : {avg_ms:.1f} ms")
    else:
        print("  Avg total latency  : N/A")
    print_separator("=")


if __name__ == "__main__":
    main()
