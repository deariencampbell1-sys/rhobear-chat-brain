from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RequestRecord:
    ts: float
    prompt_hash: str
    hit_or_miss: str
    similarity_score: float | None
    response_ms: float
    tokens_out: int
    model_used: str


class RequestLogger:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                prompt_hash TEXT NOT NULL,
                hit_or_miss TEXT NOT NULL,
                similarity_score REAL,
                response_ms REAL NOT NULL,
                tokens_out INTEGER NOT NULL,
                model_used TEXT NOT NULL
            )
            """
        )
        conn.commit()
        self._conn = conn

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("RequestLogger not connected")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def is_reachable(self) -> bool:
        if self._conn is None:
            return False
        try:
            self.conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def log(self, record: RequestRecord) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO requests (
                    ts, prompt_hash, hit_or_miss, similarity_score,
                    response_ms, tokens_out, model_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.ts,
                    record.prompt_hash,
                    record.hit_or_miss,
                    record.similarity_score,
                    record.response_ms,
                    record.tokens_out,
                    record.model_used,
                ),
            )

    def metrics(self) -> dict:
        cutoff = time.time() - 86400
        rows = self.conn.execute(
            """
            SELECT response_ms, tokens_out, hit_or_miss
            FROM requests
            WHERE ts >= ?
            ORDER BY response_ms
            """,
            (cutoff,),
        ).fetchall()

        if not rows:
            return {
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "tok_per_sec": 0.0,
                "cache_hit_rate": 0.0,
                "requests_total": 0,
                "cost_per_1k_estimate": 0.0,
            }

        response_times = [float(r[0]) for r in rows]
        tokens = [int(r[1]) for r in rows]
        hits = sum(1 for r in rows if r[2] == "hit")
        total = len(rows)

        def percentile(values: list[float], pct: float) -> float:
            if not values:
                return 0.0
            idx = max(0, min(len(values) - 1, int(round((pct / 100) * (len(values) - 1)))))
            return values[idx]

        total_tokens = sum(tokens)
        total_seconds = sum(response_times) / 1000.0
        tok_per_sec = total_tokens / total_seconds if total_seconds > 0 else 0.0

        # Estimate: cache hits ~$0, misses ~$0.0001/token on CPU llama.cpp box
        miss_tokens = sum(tokens[i] for i, r in enumerate(rows) if r[2] == "miss")
        cost_per_1k = (miss_tokens / 1000.0) * 0.0001 if miss_tokens else 0.0

        return {
            "p50_ms": round(percentile(response_times, 50), 2),
            "p95_ms": round(percentile(response_times, 95), 2),
            "tok_per_sec": round(tok_per_sec, 2),
            "cache_hit_rate": round(hits / total, 4),
            "requests_total": total,
            "cost_per_1k_estimate": round(cost_per_1k, 6),
        }