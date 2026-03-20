import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

import psycopg


def _build_connection_kwargs() -> dict:
    """
    Build keyword arguments for psycopg.connect() from environment variables.

    Supports either a single DATABASE_URL or individual DB_* variables.
    """
    database_url: Optional[str] = os.getenv("DATABASE_URL")
    if database_url:
        return {"conninfo": database_url}

    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }


def get_connection() -> psycopg.Connection:
    """
    Create and return a new psycopg connection to the Postgres database.

    Raises psycopg.OperationalError if the connection cannot be established.
    """
    kwargs = _build_connection_kwargs()
    return psycopg.connect(**kwargs)

