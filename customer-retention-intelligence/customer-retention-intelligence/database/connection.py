"""Database connection management.

Credentials come ONLY from environment variables / a local .env file (never hardcoded).
Intended backend: PostgreSQL database `customer_retention_db`.
An explicit, loudly-logged SQLite fallback exists for local development and is only used when
ALLOW_SQLITE_FALLBACK=true is set AND PostgreSQL is unreachable.
"""
from __future__ import annotations
import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
log = logging.getLogger("database")

SQLITE_FALLBACK_PATH = ROOT / "data" / "processed" / "customer_retention_fallback.db"


def build_postgres_url() -> str:
    explicit = os.getenv("DATABASE_URL")
    if explicit:
        return explicit
    missing = [k for k in ("POSTGRES_USER", "POSTGRES_PASSWORD") if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {missing}. Copy .env.example to .env and fill it in.")
    return URL.create(
        "postgresql+psycopg2",
        username=os.environ["POSTGRES_USER"], password=os.environ["POSTGRES_PASSWORD"],
        host=os.getenv("POSTGRES_HOST", "localhost"), port=int(os.getenv("POSTGRES_PORT", 5432)),
        database=os.getenv("POSTGRES_DB", "customer_retention_db"),
    ).render_as_string(hide_password=False)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    if os.getenv("FORCE_SQLITE", "").lower() == "true":
        log.warning("FORCE_SQLITE=true -> using SQLite (development only, NOT the intended PostgreSQL architecture).")
        return create_engine(f"sqlite:///{SQLITE_FALLBACK_PATH}")
    try:
        engine = create_engine(build_postgres_url(), pool_pre_ping=True)
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return engine
    except Exception as exc:  # noqa: BLE001
        if os.getenv("ALLOW_SQLITE_FALLBACK", "").lower() == "true":
            log.warning("PostgreSQL unavailable (%s). FALLING BACK TO SQLITE at %s. "
                        "This is a dev-only fallback; the intended backend is PostgreSQL.", exc, SQLITE_FALLBACK_PATH)
            return create_engine(f"sqlite:///{SQLITE_FALLBACK_PATH}")
        raise ConnectionError(
            f"Cannot connect to PostgreSQL: {exc}\nSee README 'PostgreSQL setup', or set ALLOW_SQLITE_FALLBACK=true "
            "for an explicit local-development fallback.") from exc


def backend_name() -> str:
    return get_engine().dialect.name


def check_connection() -> dict:
    eng = get_engine()
    with eng.connect() as c:
        ok = c.execute(text("SELECT 1")).scalar() == 1
    return {"ok": ok, "backend": eng.dialect.name, "database": eng.url.database}


def reset_engine_cache() -> None:
    get_engine.cache_clear()
