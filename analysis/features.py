#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm

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
    token_min_chars: int
    min_df: float | int
    max_df: float | int
    stopwords: str
    legal_refs: bool
    filter_topic_run_id: Optional[str]
    filter_topic_ids: List[int]
    filter_topic_min_weight: float


LAW_CODES = [
    "BGB", "ZPO", "GG", "BetrKV", "BGBEG", "WEG", "StGB", "StPO", "VwGO", "SGB", "HGB",
]

LAW_ALT = "|".join(sorted(LAW_CODES, key=len, reverse=True))

PARA_RE = re.compile(
    rf"§{{1,2}}\s*(\d+[a-zA-Z]?)"
    rf"(?:\s*Abs\.\s*\d+[a-zA-Z]?)?"
    rf"(?:\s*S\.\s*\d+)?"
    rf"(?:\s*Nr\.\s*\d+)?"
    rf"(?:\s*(?:des|der|dem))?"
    rf"(?:\s*({LAW_ALT}))?",
    flags=re.IGNORECASE,
)

ART_RE = re.compile(
    rf"Art\.\s*(\d+[a-zA-Z]?)"
    rf"(?:\s*Abs\.\s*\d+[a-zA-Z]?)?"
    rf"(?:\s*S\.\s*\d+)?"
    rf"(?:\s*Nr\.\s*\d+)?"
    rf"(?:\s*(?:des|der|dem))?"
    rf"\s*({LAW_ALT})",
    flags=re.IGNORECASE,
)


def _parse_topic_ids(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        v = int(token)
        if v < 0:
            raise ValueError("topic ids must be >= 0")
        out.append(v)
    if not out:
        raise ValueError("empty topic id list")
    return sorted(set(out))


def _build_sql(
    min_year: Optional[int],
    max_year: Optional[int],
    limit: Optional[int],
    filter_topic_run_id: Optional[str],
    filter_topic_ids: Optional[List[int]],
    filter_topic_min_weight: float,
) -> tuple[str, list[Any]]:
    where = ["clean_text IS NOT NULL"]
    params: list[Any] = []

    if min_year is not None:
        where.append("(decision_date IS NULL OR decision_date >= make_date(%s,1,1))")
        params.append(int(min_year))
    if max_year is not None:
        where.append("(decision_date IS NULL OR decision_date < make_date(%s,1,1))")
        params.append(int(max_year) + 1)

    if filter_topic_run_id is not None:
        if not filter_topic_ids:
            raise ValueError("topic filter requires at least one topic id")
        where.append(
            """
            EXISTS (
              SELECT 1
              FROM case_topics ct2
              WHERE ct2.case_id = cases_text.id
                AND ct2.run_id = %s
                AND ct2.topic_id = ANY(%s)
                AND ct2.weight >= %s
            )
            """
        )
        params.append(str(filter_topic_run_id))
        params.append([int(x) for x in filter_topic_ids])
        params.append(float(filter_topic_min_weight))

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
        return sorted(GERMAN_STOPWORDS)
    return None


def _extract_legal_reference_tokens(text: str) -> List[str]:
    if not text:
        return []

    out: List[str] = []

    for m in PARA_RE.finditer(text):
        para = (m.group(1) or "").strip().lower()
        law = (m.group(2) or "").strip().upper()
        if not para:
            continue

        para_clean = re.sub(r"[^0-9a-z]", "", para)
        if not para_clean:
            continue

        out.append(f"LEGREF_PAR_{para_clean}")
        if law:
            out.append(f"LEGREF_PAR_{para_clean}_{law}")

    for m in ART_RE.finditer(text):
        article = (m.group(1) or "").strip().lower()
        law = (m.group(2) or "").strip().upper()
        if not article or not law:
            continue

        article_clean = re.sub(r"[^0-9a-z]", "", article)
        if not article_clean:
            continue

        out.append(f"LEGREF_ART_{article_clean}_{law}")

    if not out:
        return out

    return sorted(set(out))


def _token_pattern(min_chars: int) -> str:
    m = max(1, int(min_chars))
    return rf"(?u)\b\w{{{m},}}\b"


def _parse_df_value(raw: str) -> float | int:
    s = str(raw).strip()
    if "." in s:
        v = float(s)
        if not (0.0 < v <= 1.0):
            raise ValueError("float min_df must be in (0, 1]")
        return v
    v = int(s)
    if v < 1:
        raise ValueError("integer min_df must be >= 1")
    return v


def _parse_max_df_value(raw: str) -> float | int:
    s = str(raw).strip()
    if "." in s:
        v = float(s)
        if not (0.0 < v <= 1.0):
            raise ValueError("float max_df must be in (0, 1]")
        return v
    v = int(s)
    if v < 1:
        raise ValueError("integer max_df must be >= 1")
    return v


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
    ap.add_argument("--token-min-chars", type=int, default=2, help="Minimum token length (default 2)")
    ap.add_argument("--min-df", default="1", help="Minimum document frequency, e.g. 5 or 0.01 (default 1)")
    ap.add_argument("--max-df", default="1.0", help="Maximum document frequency, e.g. 0.4 or 100000 (default 1.0)")
    ap.add_argument("--stopwords", default="german", help="Stopword mode: german|none")
    ap.add_argument("--legal-refs", dest="legal_refs", action="store_true", help="Extract legal references and inject canonical tokens")
    ap.add_argument("--no-legal-refs", dest="legal_refs", action="store_false", help="Disable legal reference extraction")
    ap.set_defaults(legal_refs=True)
    ap.add_argument("--filter-topic-run-id", default=None, help="Optional topic run_id to restrict cases by existing case_topics")
    ap.add_argument("--filter-topic-ids", default=None, help="Comma-separated topic ids, e.g. 2,5,9 (requires --filter-topic-run-id)")
    ap.add_argument("--filter-topic-min-weight", type=float, default=0.20, help="Minimum topic weight for filter (default 0.20)")
    ap.add_argument("--resume", action="store_true", help="Skip step if expected output artifacts already exist")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = ap.parse_args()

    t0 = time.perf_counter()

    if args.ngram_max < 1:
        raise SystemExit("--ngram-max must be >= 1")
    if args.token_min_chars < 1:
        raise SystemExit("--token-min-chars must be >= 1")
    try:
        min_df_val = _parse_df_value(args.min_df)
    except Exception as e:
        raise SystemExit(f"invalid --min-df: {e}")
    try:
        max_df_val = _parse_max_df_value(args.max_df)
    except Exception as e:
        raise SystemExit(f"invalid --max-df: {e}")

    if args.filter_topic_run_id and args.filter_topic_min_weight < 0.0:
        raise SystemExit("--filter-topic-min-weight must be >= 0")
    if args.filter_topic_run_id and args.filter_topic_min_weight > 1.0:
        raise SystemExit("--filter-topic-min-weight must be <= 1")

    topic_ids: List[int] = []
    if args.filter_topic_ids:
        try:
            topic_ids = _parse_topic_ids(args.filter_topic_ids)
        except Exception as e:
            raise SystemExit(f"invalid --filter-topic-ids: {e}")
    if args.filter_topic_run_id and not topic_ids:
        raise SystemExit("--filter-topic-run-id requires --filter-topic-ids")
    if topic_ids and not args.filter_topic_run_id:
        raise SystemExit("--filter-topic-ids requires --filter-topic-run-id")


    os.makedirs(args.out_dir, exist_ok=True)

    expected = [
        os.path.join(args.out_dir, "docs.csv"),
        os.path.join(args.out_dir, "counts.npz"),
        os.path.join(args.out_dir, "vocab.json"),
        os.path.join(args.out_dir, "config.json"),
    ]
    if args.use_tfidf:
        expected.append(os.path.join(args.out_dir, "tfidf.npz"))

    if args.resume and all(os.path.exists(p) for p in expected):
        print(f"features: resume active, artifacts already exist -> skip ({args.out_dir})")
        return

    conn = pg_connect(PgConfig())
    try:
        sql, params = _build_sql(
            args.min_year,
            args.max_year,
            args.limit,
            filter_topic_run_id=args.filter_topic_run_id,
            filter_topic_ids=topic_ids,
            filter_topic_min_weight=float(args.filter_topic_min_weight),
        )
    except Exception as e:
        conn.close()
        raise SystemExit(f"invalid topic filter config: {e}")

    docs_meta: List[Dict[str, Any]] = []
    texts: List[str] = []
    legal_ref_tokens_total = 0

    with conn.cursor() as cur:
        cur.execute(sql, params)
        fetched = cur.fetchall()
        for case_id, decision_date, clean_text in tqdm(
            fetched,
            desc="features: load docs",
            unit="doc",
            disable=args.no_progress,
        ):
            text = (clean_text or "").strip()
            if len(text) < args.min_chars:
                continue

            if args.legal_refs:
                ref_tokens = _extract_legal_reference_tokens(text)
                if ref_tokens:
                    legal_ref_tokens_total += len(ref_tokens)
                    text = text + "\n" + " ".join(ref_tokens)

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

    n_docs = len(texts)

    def _to_abs_docs(v: float | int, total_docs: int) -> float:
        if isinstance(v, float):
            return v * float(total_docs)
        return float(v)

    min_abs = _to_abs_docs(min_df_val, n_docs)
    max_abs = _to_abs_docs(max_df_val, n_docs)
    if max_abs < min_abs:
        raise SystemExit(
            f"invalid df bounds for {n_docs} docs: min_df={min_df_val} (~{min_abs:.2f} docs) > "
            f"max_df={max_df_val} (~{max_abs:.2f} docs)"
        )

    vectorizer = CountVectorizer(
        max_features=int(args.max_features),
        ngram_range=(1, int(args.ngram_max)),
        lowercase=True,
        min_df=min_df_val,
        max_df=max_df_val,
        stop_words=_select_stopwords(args.stopwords),
        token_pattern=_token_pattern(args.token_min_chars),
    )

    print("features: vectorizing counts ...")
    try:
        counts = vectorizer.fit_transform(texts)
    except ValueError as e:
        if "empty vocabulary" not in str(e):
            raise
        print("features: warning empty vocabulary with current stopword config; retrying with stopwords=none ...")
        vectorizer = CountVectorizer(
            max_features=int(args.max_features),
            ngram_range=(1, int(args.ngram_max)),
            lowercase=True,
            min_df=min_df_val,
            max_df=max_df_val,
            stop_words=None,
            token_pattern=_token_pattern(args.token_min_chars),
        )
        counts = vectorizer.fit_transform(texts)
    sparse.save_npz(os.path.join(args.out_dir, "counts.npz"), counts)

    if args.use_tfidf:
        print("features: building tfidf ...")
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
        token_min_chars=int(args.token_min_chars),
        min_df=min_df_val,
        max_df=max_df_val,
        stopwords=str(args.stopwords),
        legal_refs=bool(args.legal_refs),
        filter_topic_run_id=str(args.filter_topic_run_id) if args.filter_topic_run_id else None,
        filter_topic_ids=[int(x) for x in topic_ids],
        filter_topic_min_weight=float(args.filter_topic_min_weight),
    )

    config_obj = {
        "feature_config": asdict(cfg),
        "doc_count": len(texts),
        "vocab_size": len(vocab),
        "counts_shape": list(counts.shape),
        "tfidf_written": bool(args.use_tfidf),
        "legal_ref_tokens_total": int(legal_ref_tokens_total),
    }
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config_obj, fh, ensure_ascii=False, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"features done. docs={len(texts)} vocab={len(vocab)} out_dir={args.out_dir} elapsed_s={elapsed:.2f}")


if __name__ == "__main__":
    main()
