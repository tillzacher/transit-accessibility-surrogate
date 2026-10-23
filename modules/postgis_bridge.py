"""Worker-agnostic PostGIS connection helpers for the spatial serving store."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict
from urllib.parse import urlencode
from sqlalchemy import create_engine, text

_engine = None


def _candidate_env_paths() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    candidates: list[Path] = []
    explicit = os.environ.get("SURROGATE_POSTGIS_ENV_FILE")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(repo_root / ".env.postgis")
    candidates.append(Path.cwd() / ".env.postgis")
    return candidates


def _load_env_file() -> Dict[str, str]:
    for path in _candidate_env_paths():
        if not path.exists():
            continue
        values: Dict[str, str] = {}
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        return values
    return {}


def _settings() -> Dict[str, str]:
    file_values = _load_env_file()
    keys = {
        "url": "SURROGATE_POSTGIS_URL",
        "host": "SURROGATE_POSTGIS_HOST",
        "port": "SURROGATE_POSTGIS_PORT",
        "db": "SURROGATE_POSTGIS_DB",
        "user": "SURROGATE_POSTGIS_USER",
        "password": "SURROGATE_POSTGIS_PASSWORD",
        "sslmode": "SURROGATE_POSTGIS_SSLMODE",
        "echo": "SURROGATE_POSTGIS_SQL_ECHO",
    }
    out = {
        name: os.environ.get(env_key) or file_values.get(env_key)
        for name, env_key in keys.items()
    }
    if not out["host"]:
        out["host"] = os.environ.get("SURROGATE_POSTGIS_BIND") or file_values.get(
            "SURROGATE_POSTGIS_BIND"
        )
    if not out["port"]:
        out["port"] = "5432"
    return out


def build_postgis_url() -> str:
    settings = _settings()
    if settings["url"]:
        return settings["url"]
    missing = [
        name
        for name in ("host", "port", "db", "user", "password")
        if not settings.get(name)
    ]
    if missing:
        raise RuntimeError(
            "Missing PostGIS settings: "
            + ", ".join(missing)
            + ". Set SURROGATE_POSTGIS_URL or provide SURROGATE_POSTGIS_HOST/PORT/DB/USER/PASSWORD."
        )
    query = {}
    if settings.get("sslmode"):
        query["sslmode"] = settings["sslmode"]
    query_text = f"?{urlencode(query)}" if query else ""
    return f"postgresql+psycopg2://{settings['user']}:{settings['password']}@{settings['host']}:{settings['port']}/{settings['db']}{query_text}"


def get_engine():
    global _engine
    if _engine is not None:
        return _engine
    settings = _settings()
    sql_echo = str(settings.get("echo") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    url = build_postgis_url()
    _engine = create_engine(
        url, pool_pre_ping=True, echo=sql_echo, pool_size=32, max_overflow=16
    )
    return _engine


def reload_engine():
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None
    return get_engine()


def ping() -> bool:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("select 1"))
    return True


__all__ = ["build_postgis_url", "get_engine", "reload_engine", "ping"]
