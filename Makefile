COMPOSE_DIR := docker/postgres
COMPOSE_FILE := $(COMPOSE_DIR)/docker-compose.yml
ENV_FILE := $(COMPOSE_DIR)/.env
OPENWEBUI_COMPOSE_FILE := docker/openwebui/docker-compose.yml
OPENWEBUI_ENV_FILE := docker/openwebui/.env

-include $(ENV_FILE)
export

.PHONY: db-up db-down db-reset db-logs psql pgadmin-url etl-env db-smoke vector-enable
.PHONY: transform chunk embed counts retrieve rag rag-api openwebui-up openwebui-down openwebui-logs

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
