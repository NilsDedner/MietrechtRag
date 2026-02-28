# Case Viewer (HTML-Urteil nach ID)

Kleine FastAPI-Webseite, die auf PostgreSQL zugreift und ein Urteil anhand der `cases_raw.id` lädt.

## Start

Im Projekt-Root:

```bash
python -m uvicorn web.case_viewer.app:app --host 0.0.0.0 --port 8050 --reload
```

Dann im Browser öffnen:

- http://localhost:8050

## Voraussetzung

Die PostgreSQL-ENV-Variablen müssen gesetzt sein, z. B.:

```bash
export PGHOST=192.168.0.100
export PGPORT=5432
export PGDATABASE=mietrecht
export PGUSER=postgres
export PGPASSWORD=secret
```

PowerShell:

```powershell
$env:PGHOST="192.168.0.100"
$env:PGPORT="5432"
$env:PGDATABASE="mietrecht"
$env:PGUSER="postgres"
$env:PGPASSWORD="secret"
```

## Hinweis

Die Seite rendert `cases_raw.content_html` direkt als HTML, damit die Struktur des Urteils sichtbar bleibt.
