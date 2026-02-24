# etl/config.py
from dataclasses import dataclass
import os

@dataclass
class PgConfig:
    host: str = os.getenv("PGHOST", "localhost")
    port: int = int(os.getenv("PGPORT", "5432"))
    dbname: str = os.getenv("PGDATABASE", "")
    user: str = os.getenv("PGUSER", "")
    password: str = os.getenv("PGPASSWORD", "")
    sslmode: str = os.getenv("PGSSLMODE", "prefer")

@dataclass
class LoaderConfig:
    api_cases_url: str = "https://de.openlegaldata.io/api/cases/"
    user_agent: str = "oldp-incremental-loader/pg-4.0 (+academic use)"
    default_timeout: int = 45
