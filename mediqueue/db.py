"""SQLite-backed virtual ticket queue."""

import sqlite3
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_no     TEXT    NOT NULL,
    category      TEXT    NOT NULL CHECK (category IN ('general','followup','urgent')),
    arrived_at    REAL    NOT NULL,          -- simulation minutes since midnight
    status        TEXT    NOT NULL DEFAULT 'waiting'
                    CHECK (status IN ('waiting','serving','served')),
    predicted_wait REAL,
    service_start REAL,
    service_end   REAL,
    notified      INTEGER NOT NULL DEFAULT 0
);
"""


@dataclass
class Ticket:
    id: int
    ticket_no: str
    category: str
    arrived_at: float
    status: str
    predicted_wait: float | None
    service_start: float | None
    service_end: float | None
    notified: int


class QueueDB:
    def __init__(self, path=":memory:"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def add_ticket(self, ticket_no, category, arrived_at, predicted_wait=None) -> int:
        cur = self.conn.execute(
            "INSERT INTO tickets (ticket_no, category, arrived_at, predicted_wait)"
            " VALUES (?,?,?,?)",
            (ticket_no, category, arrived_at, predicted_wait),
        )
        self.conn.commit()
        return cur.lastrowid

    def set_status(self, ticket_id, status, service_start=None, service_end=None):
        updates = ["status=?"]
        params = [status]
        if service_start is not None:
            updates.append("service_start=?")
            params.append(service_start)
        if service_end is not None:
            updates.append("service_end=?")
            params.append(service_end)
        params.append(ticket_id)
        self.conn.execute(
            f"UPDATE tickets SET {', '.join(updates)} WHERE id=?", params
        )
        self.conn.commit()

    def mark_notified(self, ticket_id):
        self.conn.execute("UPDATE tickets SET notified=1 WHERE id=?", (ticket_id,))
        self.conn.commit()

    def waiting(self) -> list[Ticket]:
        rows = self.conn.execute(
            "SELECT * FROM tickets WHERE status='waiting' ORDER BY arrived_at, id"
        ).fetchall()
        return [Ticket(**dict(r)) for r in rows]

    def all_tickets(self) -> list[Ticket]:
        rows = self.conn.execute(
            "SELECT * FROM tickets ORDER BY arrived_at, id"
        ).fetchall()
        return [Ticket(**dict(r)) for r in rows]

    def waiting_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT category, COUNT(*) c FROM tickets"
            " WHERE status='waiting' GROUP BY category"
        ).fetchall()
        counts = {"general": 0, "followup": 0, "urgent": 0}
        for r in rows:
            counts[r["category"]] = r["c"]
        return counts

    def served_count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) c FROM tickets WHERE status='served'"
        ).fetchone()
        return row["c"]

    def close(self):
        self.conn.close()
