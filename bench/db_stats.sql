SELECT COUNT(*) AS total_chunks FROM case_chunks;

SELECT COUNT(*) AS embedded_chunks
FROM case_chunks
WHERE embedding IS NOT NULL;

SELECT
  COUNT(*) AS total_chunks,
  COUNT(*) FILTER (WHERE embedding IS NOT NULL) AS embedded_chunks,
  ROUND(
    100.0 * COUNT(*) FILTER (WHERE embedding IS NOT NULL) / NULLIF(COUNT(*), 0),
    2
  ) AS pct_embedded
FROM case_chunks;
