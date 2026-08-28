-- Materialisiert den deutschen Textvektor als gespeicherte Spalte.
--
-- Vorher wurde to_tsvector('german', chunk_text) bei jeder Rangberechnung neu
-- gebildet, was die lexikalische Suche auf rund 1.100 ms p50 gebracht hat.
-- Der GIN-Index auf der Spalte bedient danach sowohl Filterung als auch Ranking.
--
-- Aufruf:  psql ... -f sql/eval_chunks_tsv.sql

SET max_parallel_maintenance_workers = 0;  -- /dev/shm im Container ist 64 MB gross
SET maintenance_work_mem = '512MB';

ALTER TABLE eval_chunks
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('german', chunk_text)) STORED;

CREATE INDEX IF NOT EXISTS idx_eval_chunks_tsv ON eval_chunks USING gin (tsv);

-- Der alte Ausdrucksindex wird von den Queries nicht mehr benutzt.
DROP INDEX IF EXISTS idx_eval_chunks_fts;

ANALYZE eval_chunks;

SELECT pg_size_pretty(pg_total_relation_size('eval_chunks')) AS tabelle_gesamt,
       pg_size_pretty(pg_relation_size('idx_eval_chunks_tsv')) AS gin_index;
