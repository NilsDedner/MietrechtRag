#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from etl.config import PgConfig
from etl.pg_db import pg_connect

GERMAN_STOPWORDS = {
    "aber", "alle", "als", "also", "am", "an", "ander", "auch", "auf", "aus", "bei", "bin", "bis",
    "bist", "da", "dadurch", "daher", "darum", "das", "daß", "dass", "dein", "deine", "dem", "den",
    "der", "des", "dessen", "deshalb", "die", "dies", "dieser", "dieses", "doch", "dort", "du", "durch",
    "ein", "eine", "einem", "einen", "einer", "eines", "er", "es", "euer", "eure", "für", "hatte", "hatten",
    "hattest", "hattet", "hier", "hinter", "ich", "ihr", "ihre", "im", "in", "ist", "ja", "jede", "jedem",
    "jeden", "jeder", "jedes", "jener", "jenes", "jetzt", "kann", "kannst", "können", "könnt", "machen",
    "mein", "meine", "mit", "muß", "mußt", "musst", "müssen", "müßt", "nach", "nachdem", "nein", "nicht",
    "nun", "oder", "seid", "sein", "seine", "sich", "sie", "sind", "soll", "sollen", "sollst", "sollt",
    "sonst", "soweit", "sowie", "und", "unser", "unsere", "unter", "vom", "von", "vor", "wann", "warum",
    "was", "weiter", "weitere", "wenn", "wer", "werde", "werden", "werdet", "weshalb", "wie", "wieder",
    "wieso", "wir", "wird", "wirst", "wo", "woher", "wohin", "zu", "zum", "zur", "über",
}


@dataclass
class FeatureConfig:
    min_chars: int
    limit: Optional[int]
    min_year: Optional[int]
    max_year: Optional[int]
    use_tfidf: bool
    max_features: int
    ngram_max: int
    stopwords: str


def _build_sql(min_year: Optional[int], max_year: Optional[int], limit: Optional[int]) -> tuple[str, list[Any]]:
    where = ["clean_text IS NOT NULL"]
    params: list[Any] = []

    if min_year is not None:
        where.append("(decision_date IS NULL OR decision_date >= make_date(%s,1,1))")
        params.append(int(min_year))
    if max_year is not None:
        where.append("(decision_date IS NULL OR decision_date < make_date(%s,1,1))")
        params.append(int(max_year) + 1)

    sql = f"""
    SELECT id AS case_id, decision_date, clean_text
    FROM cases_text
    WHERE {' AND '.join(where)}
    ORDER BY id ASC
    """

    if limit is not None:
        sql += " LIMIT %s"
        params.append(int(limit))

    return sql, params


def _select_stopwords(name: str):
    if name.lower() == "none":
        return None
    if name.lower() == "german":
        return GERMAN_STOPWORDS
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build analysis features from cases_text")
    ap.add_argument("--out-dir", required=True, help="Output directory for artifacts")
    ap.add_argument("--min-chars", type=int, default=800, help="Min text length (default 800)")
    ap.add_argument("--limit", type=int, default=None, help="Optional maximum docs")
    ap.add_argument("--min-year", type=int, default=None, help="Optional min decision year")
    ap.add_argument("--max-year", type=int, default=None, help="Optional max decision year")
    ap.add_argument("--use-tfidf", dest="use_tfidf", action="store_true", help="Build TF-IDF matrix")
    ap.add_argument("--no-tfidf", dest="use_tfidf", action="store_false", help="Skip TF-IDF matrix")
    ap.set_defaults(use_tfidf=True)
    ap.add_argument("--max-features", type=int, default=50000, help="Max vectorizer features")
    ap.add_argument("--ngram-max", type=int, default=2, help="Max ngram range (default 2)")
    ap.add_argument("--stopwords", default="german", help="Stopword mode: german|none")
    args = ap.parse_args()

    if args.ngram_max < 1:
        raise SystemExit("--ngram-max must be >= 1")

    os.makedirs(args.out_dir, exist_ok=True)

    conn = pg_connect(PgConfig())
    sql, params = _build_sql(args.min_year, args.max_year, args.limit)

    docs_meta: List[Dict[str, Any]] = []
    texts: List[str] = []

    with conn.cursor() as cur:
        cur.execute(sql, params)
        for case_id, decision_date, clean_text in cur.fetchall():
            text = (clean_text or "").strip()
            if len(text) < args.min_chars:
                continue
            row_id = len(texts)
            texts.append(text)
            docs_meta.append(
                {
                    "row_id": row_id,
                    "case_id": int(case_id),
                    "decision_date": str(decision_date) if decision_date else "",
                    "text_len": len(text),
                }
            )

    conn.close()

    if not texts:
        raise SystemExit("No documents passed filters. Adjust --min-chars/--year range.")

    vectorizer = CountVectorizer(
        max_features=int(args.max_features),
        ngram_range=(1, int(args.ngram_max)),
        lowercase=True,
        stop_words=_select_stopwords(args.stopwords),
        token_pattern=r"(?u)\\b\\w\\w+\\b",
    )

    counts = vectorizer.fit_transform(texts)
    sparse.save_npz(os.path.join(args.out_dir, "counts.npz"), counts)

    if args.use_tfidf:
        tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False).fit_transform(counts)
        sparse.save_npz(os.path.join(args.out_dir, "tfidf.npz"), tfidf)

    with open(os.path.join(args.out_dir, "docs.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["row_id", "case_id", "decision_date", "text_len"])
        w.writeheader()
        w.writerows(docs_meta)

    vocab = {term: int(idx) for term, idx in vectorizer.vocabulary_.items()}
    with open(os.path.join(args.out_dir, "vocab.json"), "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=2)

    cfg = FeatureConfig(
        min_chars=int(args.min_chars),
        limit=args.limit,
        min_year=args.min_year,
        max_year=args.max_year,
        use_tfidf=bool(args.use_tfidf),
        max_features=int(args.max_features),
        ngram_max=int(args.ngram_max),
        stopwords=str(args.stopwords),
    )

    config_obj = {
        "feature_config": asdict(cfg),
        "doc_count": len(texts),
        "vocab_size": len(vocab),
        "counts_shape": list(counts.shape),
        "tfidf_written": bool(args.use_tfidf),
    }
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config_obj, fh, ensure_ascii=False, indent=2)

    print(f"features done. docs={len(texts)} vocab={len(vocab)} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
