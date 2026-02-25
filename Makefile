COMPOSE_DIR := docker/postgres
COMPOSE_FILE := $(COMPOSE_DIR)/docker-compose.yml
ENV_FILE := $(COMPOSE_DIR)/.env
OPENWEBUI_COMPOSE_FILE := docker/openwebui/docker-compose.yml
OPENWEBUI_ENV_FILE := docker/openwebui/.env

ANALYSIS_RUN_ID ?= analysis_$(shell date +%Y%m%d_%H%M%S)
ANALYSIS_OUT_DIR ?= artifacts/analysis/$(ANALYSIS_RUN_ID)
ANALYSIS_MIN_CHARS ?= 800
ANALYSIS_MAX_FEATURES ?= 50000
ANALYSIS_NGRAM_MAX ?= 2
ANALYSIS_STOPWORDS ?= german
ANALYSIS_N_TOPICS ?= 15
ANALYSIS_TOP_TERMS ?= 15
ANALYSIS_CLUSTER_K ?= 30
ANALYSIS_NO_PROGRESS ?= 0
ANALYSIS_RESUME ?= 0

ANALYSIS_PROGRESS_FLAG := $(if $(filter 1 true TRUE yes YES,$(ANALYSIS_NO_PROGRESS)),--no-progress,)
ANALYSIS_RESUME_FLAG := $(if $(filter 1 true TRUE yes YES,$(ANALYSIS_RESUME)),--resume,)

-include $(ENV_FILE)
export

.PHONY: db-up db-down db-reset db-logs psql pgadmin-url etl-env db-smoke vector-enable
.PHONY: transform chunk embed counts retrieve rag rag-api openwebui-up openwebui-down openwebui-logs
.PHONY: analysis-features analysis-topics analysis-cluster analysis-export analysis-all analysis-all-with-cluster
.PHONY: bench-embed bench-retrieve db-stats

db-up:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d

db-down:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) down

# ACHTUNG: löscht Volumes => Daten weg
db-reset:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) down -v
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d

db-logs:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) logs -f --tail=200 postgres

psql:
	docker exec -it mietrecht_pg psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

pgadmin-url:
	@echo "pgAdmin: http://localhost:$(PGADMIN_PORT)"

etl-env:
	@echo "export PGHOST=192.168.0.100"
	@echo "export PGPORT=$(PG_PORT)"
	@echo "export PGDATABASE=$(POSTGRES_DB)"
	@echo "export PGUSER=$(POSTGRES_USER)"
	@echo "export PGPASSWORD=$(POSTGRES_PASSWORD)"
	@echo ""
	@echo "# PowerShell"
	@echo "$${env:PGHOST}=\"192.168.0.100\""
	@echo "$${env:PGPORT}=\"$(PG_PORT)\""
	@echo "$${env:PGDATABASE}=\"$(POSTGRES_DB)\""
	@echo "$${env:PGUSER}=\"$(POSTGRES_USER)\""
	@echo "$${env:PGPASSWORD}=\"$(POSTGRES_PASSWORD)\""

db-smoke:
	docker exec -it mietrecht_pg psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -c "SELECT version();"
	docker exec -it mietrecht_pg psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -c "\dt"

vector-enable:
	docker exec -it mietrecht_pg psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -c "CREATE EXTENSION IF NOT EXISTS vector;"

# ---- ETL pipeline helpers ----

transform:
	@echo "Running transform (cases_raw -> cases_text)"
	python -m etl.transform_text --batch 2000

chunk:
	@echo "Running chunk (cases_text -> case_chunks)"
	python -m etl.chunk --batch 1000 --chunk-size 1200 --overlap 150

embed:
	@echo "Running embed (case_chunks -> pgvector embeddings)"
	python -m etl.embed_pgvector --batch 1500 --encode-batch 512 --normalize --index hnsw --only-missing

counts:
	docker exec -it mietrecht_pg psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -c "SELECT COUNT(*) AS cases_raw FROM cases_raw;"
	docker exec -it mietrecht_pg psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -c "SELECT COUNT(*) AS cases_text FROM cases_text;"
	docker exec -it mietrecht_pg psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -c "SELECT COUNT(*) AS chunks FROM case_chunks;"

retrieve:
	python -m etl.retrieve "$(Q)" --k 10 --pretty

rag:
	python -m etl.rag_answer "$(Q)" --k 10 --pretty

rag-api:
	python -m etl.rag_openai_api --host 0.0.0.0 --port 8010

openwebui-up:
	docker compose -f $(OPENWEBUI_COMPOSE_FILE) --env-file $(OPENWEBUI_ENV_FILE) up -d

openwebui-down:
	docker compose -f $(OPENWEBUI_COMPOSE_FILE) --env-file $(OPENWEBUI_ENV_FILE) down

openwebui-logs:
	docker compose -f $(OPENWEBUI_COMPOSE_FILE) --env-file $(OPENWEBUI_ENV_FILE) logs -f --tail=200

# ---- Analysis pipeline helpers ----

analysis-features:
	@echo "Running analysis.features (out=$(ANALYSIS_OUT_DIR))"
	@echo "Estimated runtime: ~2-15 min (depends on docs/CPU/RAM)"
	python -m analysis.features \
	  --out-dir "$(ANALYSIS_OUT_DIR)" \
	  --min-chars $(ANALYSIS_MIN_CHARS) \
	  --max-features $(ANALYSIS_MAX_FEATURES) \
	  --ngram-max $(ANALYSIS_NGRAM_MAX) \
	  --stopwords "$(ANALYSIS_STOPWORDS)" \
	  $(ANALYSIS_RESUME_FLAG) \
	  $(ANALYSIS_PROGRESS_FLAG)

analysis-topics:
	@echo "Running analysis.topic_modeling (run_id=$(ANALYSIS_RUN_ID))"
	@echo "Estimated runtime: ~5-40 min (depends on corpus size & n_topics)"
	python -m analysis.topic_modeling \
	  --run-id "$(ANALYSIS_RUN_ID)" \
	  --in-dir "$(ANALYSIS_OUT_DIR)" \
	  --n-topics $(ANALYSIS_N_TOPICS) \
	  --top-terms $(ANALYSIS_TOP_TERMS) \
	  $(ANALYSIS_RESUME_FLAG) \
	  $(ANALYSIS_PROGRESS_FLAG)

analysis-cluster:
	@echo "Running analysis.clustering (run_id=$(ANALYSIS_RUN_ID), k=$(ANALYSIS_CLUSTER_K))"
	@echo "Estimated runtime: ~2-20 min (depends on docs/features/k)"
	python -m analysis.clustering \
	  --run-id "$(ANALYSIS_RUN_ID)" \
	  --in-dir "$(ANALYSIS_OUT_DIR)" \
	  --k $(ANALYSIS_CLUSTER_K) \
	  $(ANALYSIS_RESUME_FLAG) \
	  $(ANALYSIS_PROGRESS_FLAG)

analysis-export:
	@echo "Running analysis.export (run_id=$(ANALYSIS_RUN_ID))"
	python -m analysis.export \
	  --run-id "$(ANALYSIS_RUN_ID)" \
	  --out-dir "$(ANALYSIS_OUT_DIR)" \
	  --include-topics \
	  --include-clusters

analysis-all: analysis-features analysis-topics analysis-export
	@echo "Analysis pipeline done. run_id=$(ANALYSIS_RUN_ID) out=$(ANALYSIS_OUT_DIR)"

analysis-all-with-cluster: analysis-features analysis-topics analysis-cluster analysis-export
	@echo "Analysis pipeline (+cluster) done. run_id=$(ANALYSIS_RUN_ID) out=$(ANALYSIS_OUT_DIR)"

# ---- Bench helpers ----

bench-embed:
	python -m bench.bench_embed --batch 1500 --encode-batch 512 --normalize --only-missing --index none

bench-retrieve:
	python -m bench.bench_retrieve --k 10 --rounds 1

db-stats:
	docker exec -i mietrecht_pg psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) < bench/db_stats.sql
