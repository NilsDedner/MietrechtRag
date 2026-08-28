#!/bin/bash
# Zentrale Umgebung für Läufe auf dem Analyse-Server.
# Verwendung:  source env.sh
#
# Liest Zugangsdaten aus den (gitignorierten) .env-Dateien und mappt sie auf die
# PG*-Variablen, die etl.config erwartet. Gibt selbst keine Geheimnisse aus.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$_here/docker/postgres/.env" ]; then
    set -a
    . "$_here/docker/postgres/.env"
    set +a
    export PGHOST="${PGHOST:-localhost}"
    export PGPORT="${PG_PORT:-5432}"
    export PGDATABASE="$POSTGRES_DB"
    export PGUSER="$POSTGRES_USER"
    export PGPASSWORD="$POSTGRES_PASSWORD"
fi

if [ -f "$_here/.env.llm" ]; then
    set -a
    . "$_here/.env.llm"
    set +a
fi

if [ -d "$_here/.venv" ]; then
    . "$_here/.venv/bin/activate"
fi

# Das Embedding-Modell wird auf die Revision festgenagelt, die die gespeicherten
# Embeddings erzeugt hat. Ohne Pinning laedt sentence-transformers eine neuere
# Revision und Query- und Dokumentvektoren stammen aus verschiedenen Modellen.
_snap="$HOME/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
if [ -d "$_snap" ]; then
    export RAG_EMBED_MODEL_PATH="$_snap"
    export HF_HUB_OFFLINE=1
fi

echo "env: PGDATABASE=${PGDATABASE:-?} PGHOST=${PGHOST:-?}:${PGPORT:-?} LLM=${RAG_LLM_MODEL:-nicht gesetzt} venv=${VIRTUAL_ENV:+aktiv}"
