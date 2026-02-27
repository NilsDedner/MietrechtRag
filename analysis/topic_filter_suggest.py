#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from typing import Dict, List


DEFAULT_KEYWORDS = [
    "miete",
    "miet",
    "mieter",
    "vermieter",
    "mietvertrag",
    "wohnung",
    "wohnraum",
    "nebenkosten",
    "betriebskosten",
    "kaution",
    "kündigung",
    "eigenbedarf",
    "mietminderung",
]


@dataclass
class TopicScore:
    topic_id: int
    score: float
    match_count: int
    matched_terms: List[str]


def _parse_keywords(raw: str) -> List[str]:
    out: List[str] = []
    for token in str(raw).split(","):
        t = token.strip().lower()
        if t:
            out.append(t)
    return sorted(set(out))


def _load_topics(path: str) -> Dict[int, List[Dict[str, object]]]:
    by_topic: Dict[int, List[Dict[str, object]]] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            topic_id = int(row["topic_id"])
            term = str(row["term"])
            weight = float(row["weight"])
            by_topic.setdefault(topic_id, []).append({"term": term, "weight": weight})
    return by_topic


def _score_topics(
    topics: Dict[int, List[Dict[str, object]]],
    keywords: List[str],
    top_terms_limit: int,
) -> List[TopicScore]:
    rows: List[TopicScore] = []

    for topic_id, terms in topics.items():
        ordered = sorted(terms, key=lambda x: float(x["weight"]), reverse=True)
        if top_terms_limit > 0:
            ordered = ordered[:top_terms_limit]

        score = 0.0
        matched_terms: List[str] = []
        for item in ordered:
            term = str(item["term"]).lower()
            weight = float(item["weight"])
            if any(k in term for k in keywords):
                score += weight
                matched_terms.append(str(item["term"]))

        rows.append(
            TopicScore(
                topic_id=int(topic_id),
                score=float(score),
                match_count=len(matched_terms),
                matched_terms=matched_terms,
            )
        )

    rows.sort(key=lambda x: (x.score, x.match_count), reverse=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Suggest topic IDs from topics.csv using keyword matching")
    ap.add_argument("--topics-csv", required=True, help="Path to topics.csv (columns: topic_id, term, weight)")
    ap.add_argument("--keywords", default=",".join(DEFAULT_KEYWORDS), help="Comma-separated keyword list")
    ap.add_argument("--top-n", type=int, default=5, help="Number of suggested topic IDs (default 5)")
    ap.add_argument("--top-terms-per-topic", type=int, default=30, help="Only score top N terms per topic (default 30)")
    ap.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold (default 0.0)")
    ap.add_argument("--emit-feature-filter-args", action="store_true", help="Print copy-paste ready args for analysis.features")
    ap.add_argument("--topic-run-id", default=None, help="Run ID used for case_topics lookup (for --emit-feature-filter-args)")
    ap.add_argument("--topic-min-weight", type=float, default=0.20, help="Weight threshold for analysis.features filter output (default 0.20)")
    ap.add_argument("--json-out", default=None, help="Optional path to write full scored result as JSON")
    args = ap.parse_args()

    if args.top_n < 1:
        raise SystemExit("--top-n must be >= 1")
    if args.top_terms_per_topic < 1:
        raise SystemExit("--top-terms-per-topic must be >= 1")
    if args.topic_min_weight < 0.0 or args.topic_min_weight > 1.0:
        raise SystemExit("--topic-min-weight must be in [0,1]")

    keywords = _parse_keywords(args.keywords)
    if not keywords:
        raise SystemExit("empty --keywords list")

    topics = _load_topics(args.topics_csv)
    if not topics:
        raise SystemExit("topics.csv contains no rows")

    scored = _score_topics(topics, keywords, int(args.top_terms_per_topic))
    filtered = [x for x in scored if x.score >= float(args.min_score) and x.match_count > 0]
    suggested = filtered[: int(args.top_n)]

    if not suggested:
        print("No topic candidates found with current keywords/threshold.")
        return

    print("Suggested topic IDs:")
    print(",".join(str(x.topic_id) for x in suggested))
    print("")
    print("Details:")
    for row in suggested:
        preview = ", ".join(row.matched_terms[:8])
        print(f"- topic_id={row.topic_id} score={row.score:.4f} matches={row.match_count} terms=[{preview}]")

    if args.emit_feature_filter_args:
        if not args.topic_run_id:
            raise SystemExit("--emit-feature-filter-args requires --topic-run-id")
        suggested_ids = ",".join(str(x.topic_id) for x in suggested)
        print("\nanalysis.features args (copy/paste):")
        print(
            f"--filter-topic-run-id {args.topic_run_id} "
            f"--filter-topic-ids {suggested_ids} "
            f"--filter-topic-min-weight {args.topic_min_weight:.2f}"
        )

    if args.json_out:
        payload = [
            {
                "topic_id": r.topic_id,
                "score": r.score,
                "match_count": r.match_count,
                "matched_terms": r.matched_terms,
            }
            for r in scored
        ]
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\nWrote full ranking to: {args.json_out}")


if __name__ == "__main__":
    main()
