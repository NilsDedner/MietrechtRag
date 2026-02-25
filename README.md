Mietrecht RAG – ETL Loader

ETL-Pipeline zum Laden von OpenLegalData-Fällen in PostgreSQL
(Modularer Loader mit Delta-Update, Resume und Dump-Import)


1. Setup

1.1 Virtuelle Umgebung erstellen (einmalig)

Im Projekt-Root:
	cd ~/mietrecht_rag
	python3 -m venv .venv
Falls venv fehlt:
	sudo apt install python3-venv

1.2 Virtuelle Umgebung aktivieren	
	source .venv/bin/activate
Du erkennst die aktive Umgebung an:
	(.venv) dedner@home-1:~/mietrecht_rag$

1.3 Abhängigkeiten installieren	
	pip install -U pip
	pip install -e .
1.4 Virtuelle Umgebung deaktivieren
	deactivate


2. Projektstruktur

mietrecht_rag/
│
├── pyproject.toml
├── README.md
└── etl/
    ├── api_cases.py
    ├── cli.py
    ├── config.py
    ├── dump_reader.py
    ├── http_client.py
    ├── load_raw_pg.py
    ├── pg_db.py
    └── timeutils.py


3. Nutzung des ETL-Tools

⚠️ Immer zuerst venv aktivieren:
	source .venv/bin/activate

3.1 Hilfe anzeigen
	python -m etl.cli --help

3.2 Dump importieren (Initialbefüllung)
	python -m etl.cli \
	  --dump /pfad/zum/cases.jsonl.gz \
	  --import-dump
Optional mit Limit (Testlauf):
	python -m etl.cli \
	  --dump /pfad/zum/cases.jsonl.gz \
	  --import-dump \
	  --import-limit 1000

3.3 Delta-Update von der API
	python -m etl.cli --server-filter --page-size 100
Das lädt nur neue oder aktualisierte Fälle basierend auf updated_date.

3.4 Vollständiger Initial-Load über API
	python -m etl.cli --full-initial-load --page-size 200

3.5 Resume / Checkpoint nutzen
Optional:
	python -m etl.cli \
	  --server-filter \
	  --checkpoint checkpoint.json

3.6 ETL -> RAG Pipeline ausführen
	python -m etl.transform_text --batch 2000
	python -m etl.chunk --batch 1000 --chunk-size 1200 --overlap 150
	python -m etl.embed_pgvector --batch 1500 --encode-batch 512 --normalize --index hnsw --only-missing

3.7 Retrieval testen
	python -m etl.retrieve "Wann ist eine Eigenbedarfskündigung wirksam?" --k 10 --pretty

3.8 Echte RAG-Antwort erzeugen (Retrieve + Generate)
	# OpenAI-kompatibler Endpoint (oder lokaler v1/chat/completions Endpoint)
	export RAG_LLM_API_URL=https://api.openai.com/v1/chat/completions
	export RAG_LLM_API_KEY=<dein_key>
	export RAG_LLM_MODEL=gpt-4o-mini
	python -m etl.rag_answer "Wann ist eine Eigenbedarfskündigung wirksam?" --k 10 --pretty

	# Nur Retrieval/Context (ohne LLM-Call)
	python -m etl.rag_answer "Frage" --k 10 --no-generate --pretty

3.9 OpenWebUI anbinden (schönes Frontend)
	# 1) Linux ENV setzen (DB + OpenAI upstream)
	export PGHOST=192.168.0.100
	export PGPORT=5432
	export PGDATABASE=mietrecht
	export PGUSER=postgres
	export PGPASSWORD=secret
	export RAG_LLM_API_URL=https://api.openai.com/v1/chat/completions
	export RAG_LLM_API_KEY=<dein_openai_key>
	export RAG_LLM_MODEL=gpt-4o-mini

	# 2) Adapter starten (OpenAI-kompatible API)
	python -m etl.rag_openai_api --host 0.0.0.0 --port 8010

	# 3) OpenWebUI -> Einstellungen -> Connections -> OpenAI API
	#    Base URL: http://<dein-server>:8010/v1
	#    API Key: beliebiger String (Adapter prüft ihn nicht)
	#    Models:
	#      - mietrecht-rag   => mit Retrieval + Quellen
	#      - openai-direct   => normales Modell (ohne RAG)
	#    Optional via ENV konfigurierbar:
	#      export RAG_API_MODEL_ID=mietrecht-rag
	#      export RAG_API_PASSTHROUGH_MODEL_ID=openai-direct
	#      export RAG_API_ENABLE_PASSTHROUGH=1

	# 4) Optional Healthcheck
	curl http://<dein-server>:8010/healthz

3.10 OpenWebUI auf Linux neu installieren (Schritt-für-Schritt)
	# 0) Docker + Compose Plugin installieren (Ubuntu/Debian)
	sudo apt update
	sudo apt install -y docker.io docker-compose-v2
	sudo systemctl enable --now docker

	# 1) Im Projekt die OpenWebUI-ENV anlegen
	cp docker/openwebui/.env.example docker/openwebui/.env

	# 2) Optional Port ändern (Default 3000)
	#    OPENWEBUI_PORT=3001 in docker/openwebui/.env setzen

	# 3) RAG Adapter starten (Host-Prozess, nicht im Container)
	#    (in separater Shell mit gesetzten PG* und RAG_LLM_* Variablen)
	python -m etl.rag_openai_api --host 0.0.0.0 --port 8010

	# 4) OpenWebUI starten
	docker compose -f docker/openwebui/docker-compose.yml --env-file docker/openwebui/.env up -d

	# 5) Logs prüfen
	docker compose -f docker/openwebui/docker-compose.yml --env-file docker/openwebui/.env logs -f --tail=100

	# 6) Browser öffnen
	#    http://<dein-linux-server>:3000

	# 7) In OpenWebUI Modell auswählen:
	#    - "mietrecht-rag" für RAG-Antworten
	#    - "openai-direct" für normalen Chat ohne Retrieval
	#    Falls nicht sichtbar: Settings -> Connections prüfen (OpenAI endpoint via compose gesetzt)

	# 8) Stoppen
	docker compose -f docker/openwebui/docker-compose.yml --env-file docker/openwebui/.env down

4. PostgreSQL Konfiguration

Die Verbindung erfolgt über ENV-Variablen:
	export PGHOST=192.168.0.100
	export PGPORT=5432
	export PGDATABASE=mietrecht
	export PGUSER=postgres
	export PGPASSWORD=secret
Optional:
	export PGSSLMODE=prefer

PowerShell (Windows):
	$env:PGHOST="192.168.0.100"
	$env:PGPORT="5432"
	$env:PGDATABASE="mietrecht"
	$env:PGUSER="postgres"
	$env:PGPASSWORD="secret"
	$env:PGSSLMODE="prefer"

5. Typischer Workflow

	cd ~/mietrecht_rag
	source .venv/bin/activate

# Delta ziehen
	python -m etl.cli --server-filter

# Danach wieder deaktivieren
	deactivate

6. Architektur-Status

Aktuell implementiert:
	Extract (Dump + API)
	Idempotenter Raw-Load nach PostgreSQL
	Delta-Resume via loader_state
	Retry / Backoff / Timeout Handling
	Transform (HTML → Text)
	Chunking
	Embedding & Vektorindex
	RAG Retrieval Layer
	RAG Answering Layer (Retrieve + Generate mit Quellen)
