#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Erzeugt den Goldstandard für die Retrieval-Evaluation.

Aus dem Evaluationskorpus werden Chunks gezogen; zu jedem Chunk formuliert ein
LLM eine Frage, die sich aus genau diesem Chunk beantworten lässt. Der Chunk ist
damit der bekannte Zieltreffer (Ground Truth).

Methodische Einschränkung, die in der Arbeit zu nennen ist: aus einem Chunk
generierte Fragen übernehmen leicht dessen Wortwahl und bevorzugen dadurch
lexikalische Verfahren. Das Prompt fordert deshalb ausdrücklich eine
Umformulierung in Alltags- bzw. Mandantensprache. Ergänzend sollte ein manuell
formuliertes Fragenset über --extra-file eingemischt werden.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

import psycopg2.extras as pgx
from tqdm import tqdm

from etl.config import PgConfig
from etl.pg_db import pg_connect
from etl.rag_answer import call_chat_completion

from .db import ensure_eval_tables, upsert_questions

SYSTEM_PROMPT = (
    "Du bist Assistent für die Erstellung juristischer Evaluationsdatensätze. "
    "Du antwortest ausschließlich mit einem JSON-Objekt, ohne Markdown-Codefence."
)

USER_TEMPLATE = """Hier ist ein Ausschnitt aus einem deutschen Gerichtsurteil:

---
{chunk}
---

Formuliere dazu EINE Frage, wie sie eine mietrechtlich betroffene Person stellen würde.

Anforderungen:
- Die Frage muss sich allein aus dem obigen Ausschnitt beantworten lassen.
- Die Frage muss ohne den Ausschnitt verständlich sein: keine Formulierungen wie "im Text", "der Ausschnitt", "vorliegend", "das Gericht" ohne Kontext.
- Keine Ja/Nein-Frage, sondern eine inhaltliche Frage.
- Formuliere in Alltagssprache und vermeide es, seltene Wortkombinationen wörtlich aus dem Ausschnitt zu übernehmen.
- Wenn der Ausschnitt keinen fassbaren rechtlichen Inhalt hat (etwa reine Formalien, Rubrum, Kostenentscheidung), setze "geeignet" auf false.

Antworte als JSON:
{{"frage": "...", "kurzantwort": "...", "geeignet": true}}"""

BAD_PHRASES = [
    "im text", "der text", "diesem text", "ausschnitt", "vorliegenden fall",
    "obigen", "dokument", "urteilstext", "hier beschrieben",
]


def _question_id(case_id: int, chunk_id: int, question: str) -> str:
    raw = f"{case_id}:{chunk_id}:{question}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _parse_json_answer(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def _acceptable(question: str) -> bool:
    q = (question or "").strip()
    if len(q) < 25 or not q.endswith("?"):
        return False
    low = q.lower()
    return not any(bad in low for bad in BAD_PHRASES)


def sample_chunks(conn, corpus_run_id: str, n: int, min_chars: int, seed: int) -> List[Dict[str, Any]]:
    with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT case_id, chunk_id, decision_date, chunk_text
            FROM eval_chunks
            WHERE length(chunk_text) >= %s
            ORDER BY case_id, chunk_id
            """,
            (int(min_chars),),
        )
        rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        raise SystemExit("eval_chunks ist leer oder enthält keine ausreichend langen Chunks")

    rng = random.Random(seed)
    # hoechstens ein Chunk je Fall, damit die Fragen nicht auf wenigen Faellen klumpen
    by_case: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        by_case.setdefault(int(r["case_id"]), []).append(r)

    cases = sorted(by_case.keys())
    rng.shuffle(cases)
    picked: List[Dict[str, Any]] = []
    for case_id in cases[: int(n)]:
        picked.append(rng.choice(by_case[case_id]))
    return picked


def generate_one(
    chunk: Dict[str, Any],
    api_url: str,
    api_key: Optional[str],
    model: str,
    max_chunk_chars: int,
) -> Optional[Dict[str, Any]]:
    text = (chunk["chunk_text"] or "")[:max_chunk_chars]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(chunk=text)},
    ]
    try:
        raw = call_chat_completion(
            api_url=api_url,
            api_key=api_key,
            model=model,
            messages=messages,
            timeout=90,
            temperature=0.4,
            max_retries=4,
            initial_backoff=2.0,
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    parsed = _parse_json_answer(raw)
    if not parsed:
        return {"error": "unparsbare LLM-Antwort"}
    if not parsed.get("geeignet", True):
        return None

    question = str(parsed.get("frage") or "").strip()
    if not _acceptable(question):
        return None

    return {
        "question_id": _question_id(int(chunk["case_id"]), int(chunk["chunk_id"]), question),
        "question": question,
        "gold_case_id": int(chunk["case_id"]),
        "gold_chunk_id": int(chunk["chunk_id"]),
        "short_answer": str(parsed.get("kurzantwort") or "").strip(),
        "chunk_chars": len(chunk["chunk_text"] or ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Goldstandard-Fragen aus dem Evaluationskorpus erzeugen")
    ap.add_argument("--corpus-run-id", default="analysis_night_20260225_mietrecht_v9")
    ap.add_argument("--n", type=int, default=60, help="Anzahl zu ziehender Chunks (Default 60)")
    ap.add_argument("--min-chars", type=int, default=900, help="Mindestlänge eines Quellchunks")
    ap.add_argument("--max-chunk-chars", type=int, default=3000, help="Maximal an das LLM gesendete Zeichen")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6, help="Parallele LLM-Aufrufe")
    ap.add_argument("--extra-file", default=None, help="JSONL mit manuell erstellten Fragen zum Einmischen")
    ap.add_argument("--out-dir", default="artifacts/eval")
    ap.add_argument("--llm-api-url", default=os.getenv("RAG_LLM_API_URL", "https://api.openai.com/v1/chat/completions"))
    ap.add_argument("--llm-api-key", default=os.getenv("RAG_LLM_API_KEY"))
    ap.add_argument("--llm-model", default=os.getenv("RAG_LLM_MODEL", "gpt-4o-mini"))
    ap.add_argument("--no-db-write", dest="db_write", action="store_false")
    ap.set_defaults(db_write=True)
    args = ap.parse_args()

    if not args.llm_api_key:
        raise SystemExit("RAG_LLM_API_KEY fehlt (source env.sh)")

    t0 = time.perf_counter()
    conn = pg_connect(PgConfig())
    ensure_eval_tables(conn)

    chunks = sample_chunks(conn, args.corpus_run_id, args.n, args.min_chars, args.seed)
    print(f"goldstandard: {len(chunks)} Chunks gezogen, generiere Fragen mit {args.llm_model} ...")

    accepted: List[Dict[str, Any]] = []
    rejected = 0
    errors: List[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {
            pool.submit(generate_one, c, args.llm_api_url, args.llm_api_key, args.llm_model, args.max_chunk_chars): c
            for c in chunks
        }
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), unit="frage"):
            result = fut.result()
            if result is None:
                rejected += 1
            elif "error" in result:
                errors.append(result["error"])
            else:
                accepted.append(result)

    if args.extra_file:
        with open(args.extra_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj.setdefault(
                    "question_id",
                    _question_id(int(obj["gold_case_id"]), int(obj["gold_chunk_id"]), obj["question"]),
                )
                obj["_manual"] = True
                accepted.append(obj)

    if not accepted:
        raise SystemExit("keine verwertbaren Fragen erzeugt")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"goldstandard_{args.corpus_run_id}.jsonl")
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in accepted:
            row["source"] = "manuell" if row.get("_manual") else f"llm:{args.llm_model}"
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.db_write:
        db_rows = [
            {
                "question_id": r["question_id"],
                "corpus_run_id": args.corpus_run_id,
                "question": r["question"],
                "gold_case_id": int(r["gold_case_id"]),
                "gold_chunk_id": int(r["gold_chunk_id"]),
                "source": "manuell" if r.get("_manual") else f"llm:{args.llm_model}",
                "meta_json": pgx.Json({
                    "short_answer": r.get("short_answer", ""),
                    "chunk_chars": r.get("chunk_chars"),
                    "seed": args.seed,
                }),
            }
            for r in accepted
        ]
        upsert_questions(conn, db_rows)
        conn.commit()

    conn.close()
    print(
        f"goldstandard fertig. akzeptiert={len(accepted)} verworfen={rejected} fehler={len(errors)} "
        f"datei={out_path} elapsed_s={time.perf_counter() - t0:.1f}"
    )
    if errors:
        print("erste Fehler:", errors[:3])


if __name__ == "__main__":
    main()
