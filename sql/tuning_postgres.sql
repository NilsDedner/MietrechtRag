-- Serverkonfiguration für den Analyse-Server (46 GB RAM, 20 Kerne, SSD).
-- Ausgangszustand waren die Werkseinstellungen (shared_buffers 128 MB, work_mem 4 MB).
-- Aufruf:  psql ... -f sql/tuning_postgres.sql
-- shared_buffers wirkt erst nach einem Neustart des Containers.

ALTER SYSTEM SET shared_buffers = '12GB';
ALTER SYSTEM SET effective_cache_size = '32GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET work_mem = '32MB';
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

SELECT pg_reload_conf();

SELECT name, setting, unit, pending_restart
FROM pg_settings
WHERE name IN (
    'shared_buffers', 'work_mem', 'effective_cache_size', 'maintenance_work_mem',
    'max_parallel_workers_per_gather', 'random_page_cost', 'effective_io_concurrency'
)
ORDER BY name;
