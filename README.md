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

4. PostgreSQL Konfiguration

Die Verbindung erfolgt über ENV-Variablen:
	export PGHOST=localhost
	export PGPORT=5432
	export PGDATABASE=mietrecht
	export PGUSER=postgres
	export PGPASSWORD=secret
Optional:
	export PGSSLMODE=prefer

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
	Nächster Schritt im Projekt:
	Transform (HTML → Text)
	Chunking
	Embedding & Vektorindex
	RAG Retrieval Layer
