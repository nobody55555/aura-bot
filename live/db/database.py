"""
SQLite persistence layer for Aura live trading.

Simple, zero-configuration, WAL mode enabled.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path("data/aura_live.db")


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS orders (
    client_order_id   TEXT PRIMARY KEY,
    exchange_order_id TEXT,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL,
    type              TEXT NOT NULL,
    amount            REAL NOT NULL,
    price             REAL,
    stop_price        REAL,
    status            TEXT NOT NULL,
    filled            REAL DEFAULT 0,
    remaining         REAL,
    average_price     REAL,
    fee               REAL DEFAULT 0,
    fee_currency      TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    params            TEXT,
    parent_order_id   TEXT,
    strategy_tag      TEXT,
    error_message     TEXT
);

CREATE TABLE IF NOT EXISTS fills (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id          TEXT NOT NULL,
    exchange_fill_id  TEXT,
    price             REAL NOT NULL,
    amount            REAL NOT NULL,
    fee               REAL DEFAULT 0,
    fee_currency      TEXT,
    timestamp         TEXT NOT NULL,
    FOREIGN KEY(order_id) REFERENCES orders(client_order_id)
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    timestamp         TEXT PRIMARY KEY,
    total_equity      REAL NOT NULL,
    available_balance REAL NOT NULL,
    unrealized_pnl    REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS risk_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,
    event_type        TEXT NOT NULL,
    details           TEXT
);

CREATE TABLE IF NOT EXISTS system_state (
    key               TEXT PRIMARY KEY,
    value             TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(order_id);
"""


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Create tables if they do not exist."""
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
