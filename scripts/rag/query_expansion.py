"""UCSD-domain query expansion using a term synonym dictionary.

Detects matching keys in the query (word-boundary aware, case-insensitive)
and appends their expansion terms to the query string.

Usage (as module):
    from query_expansion import QueryExpander
    expander = QueryExpander()
    result = expander.expand("Can I take fewer units as an F-1 student?")
    # result.original_query  → original text
    # result.expanded_query  → text + appended expansion terms
    # result.matched_keys    → keys that fired

Usage (CLI):
    python scripts/rag/query_expansion.py \
        --query "Can I take fewer units as an F-1 student?" \
        --config configs/rag_query_expansion.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG = Path("configs/rag_query_expansion.json")


@dataclass
class ExpansionResult:
    original_query: str
    expanded_query: str
    matched_keys: list[str] = field(default_factory=list)
    appended_terms: list[str] = field(default_factory=list)

    @property
    def did_expand(self) -> bool:
        return bool(self.matched_keys)


def _word_boundary_pattern(phrase: str) -> re.Pattern:
    """Return a compiled regex that matches phrase at non-alphanumeric boundaries."""
    return re.compile(
        r"(?<![a-zA-Z0-9])" + re.escape(phrase) + r"(?![a-zA-Z0-9])",
        re.IGNORECASE,
    )


class QueryExpander:
    def __init__(self, config_path: Path = DEFAULT_CONFIG) -> None:
        raw: dict[str, list[str]] = json.loads(Path(config_path).read_text())
        # Store (original_key, expansion_terms, compiled_pattern) tuples
        # Sort by key length descending so longer/more specific keys match first
        self._entries = sorted(
            [
                (key, terms, _word_boundary_pattern(key))
                for key, terms in raw.items()
            ],
            key=lambda t: len(t[0]),
            reverse=True,
        )

    def expand(self, query: str) -> ExpansionResult:
        matched_keys: list[str] = []
        appended_terms: list[str] = []
        seen_terms: set[str] = set(query.lower().split())

        for key, terms, pattern in self._entries:
            if pattern.search(query):
                matched_keys.append(key)
                for term in terms:
                    if term.lower() not in seen_terms:
                        appended_terms.append(term)
                        # Prevent duplicate expansion from multiple key matches
                        seen_terms.update(term.lower().split())

        expanded_query = (
            query + " " + " ".join(appended_terms) if appended_terms else query
        )
        return ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            matched_keys=matched_keys,
            appended_terms=appended_terms,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand a query using UCSD synonym dictionary.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expander = QueryExpander(config_path=Path(args.config))
    result = expander.expand(args.query)
    print(f"Original:  {result.original_query}")
    print(f"Expanded:  {result.expanded_query}")
    if result.matched_keys:
        print(f"Matched:   {result.matched_keys}")
        print(f"Appended:  {result.appended_terms}")
    else:
        print("No expansion applied.")


if __name__ == "__main__":
    main()
