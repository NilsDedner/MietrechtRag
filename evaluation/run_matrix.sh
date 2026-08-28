#!/bin/bash
# Wartet auf einen laufenden Embedding-Job und fährt danach die komplette
# Messreihe. Gedacht für nohup/setsid, damit die Kette ohne Sitzung weiterläuft.
set -u
cd "$HOME/mietrecht_rag" || exit 1
source env.sh >/dev/null 2>&1

echo "== warte auf embed_corpus =="
while pgrep -f "evaluation.embed_corpus" >/dev/null; do sleep 30; done
echo "== embed_corpus beendet: $(date +%H:%M:%S) =="

echo "== Volltextspalte materialisieren =="
psql -q -f sql/eval_chunks_tsv.sql

echo "== Messreihe Retrieval =="
python -u -m evaluation.run_eval --run-id m_dense_minilm  --variant dense   --k 10 --ef-search 200 \
    --model "$RAG_EMBED_MODEL_PATH" 2>&1 | grep -E "hit@|mrr|ndcg|latenz"

python -u -m evaluation.run_eval --run-id m_lexical       --variant lexical --k 10 \
    --model "$RAG_EMBED_MODEL_PATH" 2>&1 | grep -E "hit@|mrr|ndcg|latenz"

python -u -m evaluation.run_eval --run-id m_hybrid_minilm --variant hybrid  --k 10 --ef-search 200 \
    --model "$RAG_EMBED_MODEL_PATH" 2>&1 | grep -E "hit@|mrr|ndcg|latenz"

python -u -m evaluation.run_eval --run-id m_dense_e5      --variant dense   --k 10 --ef-search 200 \
    --model intfloat/multilingual-e5-base --embedding-column embedding_e5 --query-prefix "query: " \
    2>&1 | grep -E "hit@|mrr|ndcg|latenz"

python -u -m evaluation.run_eval --run-id m_hybrid_e5     --variant hybrid  --k 10 --ef-search 200 \
    --model intfloat/multilingual-e5-base --embedding-column embedding_e5 --query-prefix "query: " \
    2>&1 | grep -E "hit@|mrr|ndcg|latenz"

echo "== Messreihe fertig: $(date +%H:%M:%S) =="
