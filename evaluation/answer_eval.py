#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bewertet die generierte Antwort, nicht nur das Retrieval.

Zwei Bedingungen je Frage:
  rag    - Antwort auf Basis der abgerufenen Chunks
  norag  - dasselbe Modell ohne jeden Kontext

Ein zweiter LLM-Aufruf (Judge) bewertet beide Antworten gegen die im
Goldstandard hinterlegte Kurzantwort:
  korrektheit  0..2  inhaltliche Übereinstimmung mit der Referenz
  fundierung   0..2  nur rag: Deckung durch den mitgelieferten Kontext
                     (bei norag als nicht anwendbar gewertet)
  erfunden     bool  Behauptungen ohne Beleg im Kontext

Beispiel:
    python -m evaluation.answer_eval --run-id ans_hybrid --variant hybrid --k 5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import psycopg2.extras as pgx
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from etl.config import PgConfig
from etl.pg_db import pg_connect
from etl.rag_answer import call_chat_completion

from .db import ensure_eval_tables, insert_run
from .retrievers import dense_search, fetch_chunk_texts, hybrid_search, lexical_search, vec_to_pgvector_str

ANSWER_SYSTEM = (
    "Du bist ein juristischer Assistent für deutsches Mietrecht. "
    "Antworte knapp und präzise in höchstens fünf Sätzen."
)

RAG_SYSTEM = ANSWER_SYSTEM + (
    " Stütze dich ausschließlich auf die bereitgestellten Quellen und zitiere mit [S1], [S2]. "
    "Reichen die Quellen nicht aus, sage das ausdrücklich."
)

JUDGE_SYSTEM = (
    "Du bewertest Antworten auf mietrechtliche Fragen. Du urteilst streng und "
    "antwortest ausschließlich mit einem JSON-Objekt, ohne Markdown."
)

JUDGE_TEMPLATE = """Frage:
{question}

Referenzantwort (aus dem Urteil, gilt als korrekt):
{reference}

Zu bewertende Antwort:
{answer}

{context_block}
Bewerte:
- "korrektheit": 0 (widerspricht der Referenz oder geht am Thema vorbei), 1 (teilweise richtig), 2 (deckt die Referenz inhaltlich ab)
- "fundierung": 0 (nicht durch den Kontext gedeckt), 1 (teilweise gedeckt), 2 (vollständig gedeckt); ohne Kontext immer null
- "erfunden": true, wenn konkrete Behauptungen aufgestellt werden, die weder Referenz noch Kontext hergeben

Antworte als JSON: {{"korrektheit": <0|1|2>, "fundierung": <0|1|2|null>, "erfunden": <true|false>, "begruendung": "<ein Satz>"}}"""


def load_questions(conn, corpus_run_id: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    sql = """
    SELECT question_id, question, gold_case_id, gold_chunk_id, meta_json
    FROM eval_questions
    WHERE corpus_run_id = %s
    ORDER BY question_id
    """
    params: List[Any] = [corpus_run_id]
    if limit:
        sql += " LIMIT %s"
        params.append(int(limit))
    with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def build_context(texts: List[str], max_chars: int) -> str:
    blocks: List[str] = []
    used = 0
    for i, text in enumerate(texts, start=1):
        block = f"[S{i}] {text.strip()}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def parse_judge(raw: str) -> Dict[str, Any]:
    txt = (raw or "").strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1] if "```" in txt[3:] else txt.strip("`")
        txt = txt.removeprefix("json").strip()
    start, end = txt.find("{"), txt.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"kein JSON in Judge-Antwort: {raw[:120]}")
    return json.loads(txt[start:end + 1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Antwortqualität mit und ohne Retrieval bewerten")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--variant", default="hybrid", choices=["dense", "lexical", "hybrid"])
    ap.add_argument("--corpus-run-id", default="analysis_night_20260225_mietrecht_v9")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ef-search", type=int, default=200)
    ap.add_argument("--embedding-column", default="embedding")
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--max-context-chars", type=int, default=8000)
    ap.add_argument("--model", default=os.getenv("RAG_EMBED_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out-dir", default="artifacts/eval")
    ap.add_argument("--llm-api-url", default=os.getenv("RAG_LLM_API_URL", "https://api.openai.com/v1/chat/completions"))
    ap.add_argument("--llm-api-key", default=os.getenv("RAG_LLM_API_KEY"))
    ap.add_argument("--llm-model", default=os.getenv("RAG_LLM_MODEL", "gpt-4o-mini"))
    ap.add_argument("--judge-model", default=os.getenv("RAG_JUDGE_MODEL", "gpt-4o-mini"))
    ap.add_argument("--no-db-write", dest="db_write", action="store_false")
    ap.set_defaults(db_write=True)
    args = ap.parse_args()

    if not args.llm_api_key:
        raise SystemExit("RAG_LLM_API_KEY fehlt (source .env.llm)")

    t0 = time.perf_counter()
    conn = pg_connect(PgConfig())
    ensure_eval_tables(conn)
    questions = load_questions(conn, args.corpus_run_id, args.limit)
    print(f"answer_eval: {len(questions)} Fragen, Retrieval={args.variant}, k={args.k}")

    model = SentenceTransformer(args.model)
    q_vecs = [
        vec_to_pgvector_str(v)
        for v in model.encode(
            [args.query_prefix + q["question"] for q in questions],
            convert_to_numpy=True, normalize_embeddings=True, batch_size=32,
        )
    ]

    contexts: List[str] = []
    for i, q in enumerate(tqdm(questions, desc="retrieval", unit="frage")):
        if args.variant == "dense":
            hits = dense_search(conn, q_vecs[i], args.k, ef_search=args.ef_search, column=args.embedding_column)
        elif args.variant == "lexical":
            hits = lexical_search(conn, q["question"], args.k)
        else:
            hits = hybrid_search(conn, q["question"], q_vecs[i], args.k,
                                 ef_search=args.ef_search, column=args.embedding_column)
        texts = fetch_chunk_texts(conn, hits)
        ordered = [texts.get((c, ch), "") for c, ch, _ in hits]
        contexts.append(build_context(ordered, args.max_context_chars))
    conn.close()

    def ask(payload: Tuple[int, str]) -> Tuple[int, str, str]:
        idx, condition = payload
        q = questions[idx]["question"]
        if condition == "rag":
            messages = [
                {"role": "system", "content": RAG_SYSTEM},
                {"role": "user", "content": f"Frage:\n{q}\n\nQuellen:\n{contexts[idx]}"},
            ]
        else:
            messages = [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": f"Frage:\n{q}"},
            ]
        answer = call_chat_completion(
            api_url=args.llm_api_url, api_key=args.llm_api_key, model=args.llm_model,
            messages=messages, timeout=90, temperature=args.temperature,
            max_retries=4, initial_backoff=2.0,
        )
        return idx, condition, answer

    jobs = [(i, cond) for i in range(len(questions)) for cond in ("rag", "norag")]
    answers: Dict[Tuple[int, str], str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for idx, cond, answer in tqdm(pool.map(ask, jobs), total=len(jobs), desc="antworten", unit="call"):
            answers[(idx, cond)] = answer

    def judge(payload: Tuple[int, str]) -> Tuple[int, str, Dict[str, Any]]:
        idx, condition = payload
        meta = questions[idx].get("meta_json") or {}
        reference = meta.get("short_answer") or "(keine Referenzantwort hinterlegt)"
        context_block = f"Kontext, der der Antwort vorlag:\n{contexts[idx]}\n\n" if condition == "rag" else \
                        "Der Antwort lag kein Kontext vor, bewerte \"fundierung\" als null.\n\n"
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_TEMPLATE.format(
                question=questions[idx]["question"], reference=reference,
                answer=answers[(idx, condition)], context_block=context_block)},
        ]
        raw = call_chat_completion(
            api_url=args.llm_api_url, api_key=args.llm_api_key, model=args.judge_model,
            messages=messages, timeout=90, temperature=0.0, max_retries=4, initial_backoff=2.0,
        )
        try:
            return idx, condition, parse_judge(raw)
        except Exception as e:
            return idx, condition, {"error": str(e)}

    verdicts: Dict[Tuple[int, str], Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for idx, cond, verdict in tqdm(pool.map(judge, jobs), total=len(jobs), desc="bewertung", unit="call"):
            verdicts[(idx, cond)] = verdict

    metrics: Dict[str, Any] = {}
    for cond in ("rag", "norag"):
        korrekt = [v["korrektheit"] for (i, c), v in verdicts.items() if c == cond and "korrektheit" in v]
        # Fundierung ist nur mit Kontext definiert; manche Judge-Antworten liefern
        # für norag trotz Anweisung eine 0 statt null.
        fundiert = [v["fundierung"] for (i, c), v in verdicts.items()
                    if c == cond == "rag" and isinstance(v.get("fundierung"), (int, float))]
        erfunden = [bool(v.get("erfunden")) for (i, c), v in verdicts.items() if c == cond and "erfunden" in v]
        metrics[f"{cond}_korrektheit_mittel"] = round(statistics.mean(korrekt), 3) if korrekt else None
        metrics[f"{cond}_korrekt_anteil"] = round(sum(1 for x in korrekt if x == 2) / len(korrekt), 3) if korrekt else None
        metrics[f"{cond}_fundierung_mittel"] = round(statistics.mean(fundiert), 3) if fundiert else None
        metrics[f"{cond}_erfunden_anteil"] = round(sum(erfunden) / len(erfunden), 3) if erfunden else None
        metrics[f"{cond}_bewertet"] = len(korrekt)

    params = {
        "variant": args.variant, "k": args.k, "ef_search": args.ef_search,
        "embedding_column": args.embedding_column, "embedding_model": args.model,
        "llm_model": args.llm_model, "judge_model": args.judge_model,
        "temperature": args.temperature, "corpus_run_id": args.corpus_run_id,
    }

    detail = [
        {
            "question_id": questions[i]["question_id"],
            "question": questions[i]["question"],
            "reference": (questions[i].get("meta_json") or {}).get("short_answer"),
            "rag_answer": answers.get((i, "rag")),
            "norag_answer": answers.get((i, "norag")),
            "rag_verdict": verdicts.get((i, "rag")),
            "norag_verdict": verdicts.get((i, "norag")),
        }
        for i in range(len(questions))
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.run_id}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"run_id": args.run_id, "params": params, "metrics": metrics, "per_question": detail},
                  fh, ensure_ascii=False, indent=2)

    if args.db_write:
        conn = pg_connect(PgConfig())
        insert_run(conn, run_id=args.run_id, pipeline="answer_eval", params=params, metrics=metrics)
        conn.commit()
        conn.close()

    print(f"\n=== {args.run_id} ===")
    for key, value in metrics.items():
        print(f"  {key:<28} {value}")
    print(f"Artefakt: {out_path}  (elapsed_s={time.perf_counter() - t0:.1f})")


if __name__ == "__main__":
    main()
