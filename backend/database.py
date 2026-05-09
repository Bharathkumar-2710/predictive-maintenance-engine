"""
Database Layer — SQLite with MySQL-compatible DDL schema.
Stores sensor logs, machine health scores, and alert history.
Exposes a clean DatabaseManager class following the Repository pattern.
"""

import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime
from typing import Optional
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "predictive_maintenance.db")

# ──────────────────────────────────────────────
# Schema (MySQL-compatible DDL comments included)
# ──────────────────────────────────────────────

DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS machines (
        machine_id      TEXT PRIMARY KEY,          -- VARCHAR(20) in MySQL
        registered_at   TEXT NOT NULL,             -- DATETIME in MySQL
        max_cycle       INTEGER NOT NULL,
        degradation_rate REAL NOT NULL,
        is_active       INTEGER NOT NULL DEFAULT 1 -- TINYINT(1) / BOOL in MySQL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sensor_logs (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,   -- AUTO_INCREMENT in MySQL
        machine_id      TEXT NOT NULL,
        timestamp       TEXT NOT NULL,
        cycle           INTEGER NOT NULL,
        health_score    REAL NOT NULL,
        rul             INTEGER NOT NULL,
        alert_level     TEXT NOT NULL,
        temperature     REAL,
        vibration       REAL,
        pressure        REAL,
        rpm             REAL,
        oil_viscosity   REAL,
        current_draw    REAL,
        FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
        -- In MySQL: INDEX idx_machine_cycle (machine_id, cycle)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sensor_logs_machine_cycle
        ON sensor_logs (machine_id, cycle)
    """,
    """
    CREATE TABLE IF NOT EXISTS health_snapshots (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        machine_id      TEXT NOT NULL,
        snapshot_at     TEXT NOT NULL,
        cycle           INTEGER NOT NULL,
        health_score    REAL NOT NULL,
        predicted_rul   INTEGER,
        actual_rul      INTEGER,
        alert_level     TEXT NOT NULL,
        days_to_failure REAL,
        FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS alerts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        machine_id  TEXT NOT NULL,
        timestamp   TEXT NOT NULL,
        severity    TEXT NOT NULL,
        sensor_name TEXT NOT NULL,
        value       REAL NOT NULL,
        threshold   REAL NOT NULL,
        message     TEXT NOT NULL,
        acknowledged INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
    )
    """,
]


# ──────────────────────────────────────────────
# DatabaseManager
# ──────────────────────────────────────────────

class DatabaseManager:
    """
    Repository-pattern database manager.
    Thread-safe via per-call connection creation (appropriate for Flask with SQLite).
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # better concurrent reads
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            for stmt in DDL_STATEMENTS:
                conn.execute(stmt)

    # ── Machines ───────────────────────────────

    def upsert_machine(self, machine_id: str, max_cycle: int,
                       degradation_rate: float) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO machines
                    (machine_id, registered_at, max_cycle, degradation_rate, is_active)
                VALUES (?, ?, ?, ?, 1)
            """, (machine_id, datetime.utcnow().isoformat(), max_cycle, degradation_rate))

    def list_machines(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM machines ORDER BY machine_id").fetchall()
            return [dict(r) for r in rows]

    # ── Sensor Logs ────────────────────────────

    def insert_sensor_log(self, row: dict) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO sensor_logs
                    (machine_id, timestamp, cycle, health_score, rul, alert_level,
                     temperature, vibration, pressure, rpm, oil_viscosity, current_draw)
                VALUES (:machine_id, :timestamp, :cycle, :health_score, :rul, :alert_level,
                        :temperature, :vibration, :pressure, :rpm, :oil_viscosity, :current_draw)
            """, row)

    def bulk_insert_sensor_logs(self, df: pd.DataFrame) -> int:
        """Efficient bulk insert from a DataFrame. Returns rows inserted."""
        cols = ["machine_id", "timestamp", "cycle", "health_score", "rul", "alert_level",
                "temperature", "vibration", "pressure", "rpm", "oil_viscosity", "current_draw"]
        records = df[cols].to_dict(orient="records")
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO sensor_logs
                    (machine_id, timestamp, cycle, health_score, rul, alert_level,
                     temperature, vibration, pressure, rpm, oil_viscosity, current_draw)
                VALUES (:machine_id, :timestamp, :cycle, :health_score, :rul, :alert_level,
                        :temperature, :vibration, :pressure, :rpm, :oil_viscosity, :current_draw)
            """, records)
        return len(records)

    def get_machine_history(self, machine_id: str, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM sensor_logs
                WHERE machine_id = ?
                ORDER BY cycle DESC LIMIT ?
            """, (machine_id, limit)).fetchall()
            return [dict(r) for r in rows]

    def get_all_logs(self) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query("SELECT * FROM sensor_logs ORDER BY machine_id, cycle", conn)

    # ── Health Snapshots ───────────────────────

    def upsert_health_snapshot(self, data: dict) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO health_snapshots
                    (machine_id, snapshot_at, cycle, health_score, predicted_rul,
                     actual_rul, alert_level, days_to_failure)
                VALUES (:machine_id, :snapshot_at, :cycle, :health_score, :predicted_rul,
                        :actual_rul, :alert_level, :days_to_failure)
            """, data)

    def get_latest_snapshots(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT h.* FROM health_snapshots h
                INNER JOIN (
                    SELECT machine_id, MAX(id) AS max_id
                    FROM health_snapshots GROUP BY machine_id
                ) latest ON h.machine_id = latest.machine_id AND h.id = latest.max_id
                ORDER BY h.health_score ASC
            """).fetchall()
            return [dict(r) for r in rows]

    # ── Alerts ─────────────────────────────────

    def insert_alert(self, alert_dict: dict) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO alerts
                    (machine_id, timestamp, severity, sensor_name, value, threshold, message)
                VALUES (:machine_id, :timestamp, :severity, :sensor_name, :value, :threshold, :message)
            """, {**alert_dict, "timestamp": datetime.utcnow().isoformat()})

    def get_recent_alerts(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM alerts
                ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_alert_summary(self) -> dict:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT severity, COUNT(*) as count
                FROM alerts GROUP BY severity
            """).fetchall()
            return {r["severity"]: r["count"] for r in rows}

    # ── Analytics Queries ─────────────────────

    def get_at_risk_machines(self, health_threshold: float = 0.5) -> list[dict]:
        """Return machines with health score below threshold from latest snapshots."""
        snaps = self.get_latest_snapshots()
        return [s for s in snaps if s["health_score"] <= health_threshold]

    def get_health_trend(self, machine_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT cycle, health_score, alert_level, rul
                FROM sensor_logs
                WHERE machine_id = ?
                ORDER BY cycle ASC
            """, (machine_id,)).fetchall()
            return [dict(r) for r in rows]

    def summary_stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM sensor_logs").fetchone()[0]
            machines = conn.execute("SELECT COUNT(*) FROM machines").fetchone()[0]
            alerts   = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            critical = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE severity='CRITICAL'"
            ).fetchone()[0]
            return {
                "total_readings": total,
                "total_machines": machines,
                "total_alerts":   alerts,
                "critical_alerts": critical,
            }


# Module-level singleton
_db: Optional[DatabaseManager] = None

def get_db() -> DatabaseManager:
    global _db
    if _db is None:
        _db = DatabaseManager()
    return _db
