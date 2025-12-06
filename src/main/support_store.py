"""Lightweight SQLite-backed store for support tickets."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path("data/support.db")


def init_db() -> None:
    """Ensure the database and table exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS support_tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_number TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL,
                message TEXT NOT NULL
            )
            """
        )


def create_ticket(ticket_number: str, message: str, status: str = "Unresolved") -> None:
    """Insert or replace a ticket record."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO support_tickets (ticket_number, status, message)
            VALUES (?, ?, ?)
            """,
            (ticket_number, status, message),
        )


def get_ticket_status(ticket_number: str) -> Optional[str]:
    """Return the status for a ticket number, or None if not found."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT status FROM support_tickets WHERE ticket_number = ?",
            (ticket_number,),
        ).fetchone()
    return row[0] if row else None

