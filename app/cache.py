from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec


def serialize_f32(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


@dataclass
class CacheMatch:
    question_hash: str
    question: str
    answer: str
    thought: str | None
    similarity: float


class SemanticCache:
    def __init__(self, db_path: Path, embedding_dim: int) -> None:
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS cache_vectors USING vec0(
                embedding float[{self.embedding_dim}] distance_metric=cosine
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY,
                question_hash TEXT UNIQUE NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                thought TEXT
            )
            """
        )
        conn.commit()
        self._conn = conn

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SemanticCache not connected")
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

    def has_hash(self, question_hash: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM cache_entries WHERE question_hash = ?",
            (question_hash,),
        ).fetchone()
        return row is not None

    def count_entries(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()
        return int(row[0]) if row else 0

    def insert_entry(
        self,
        question_hash: str,
        question: str,
        answer: str,
        thought: str | None,
        embedding: list[float],
    ) -> bool:
        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT OR IGNORE INTO cache_entries (question_hash, question, answer, thought)
                VALUES (?, ?, ?, ?)
                """,
                (question_hash, question, answer, thought),
            )
            if cursor.rowcount == 0:
                return False
            # Fetch the row id explicitly; cursor.lastrowid can be unreliable on
            # the OR IGNORE code path across sqlite driver versions.
            row = self.conn.execute(
                "SELECT id FROM cache_entries WHERE question_hash = ?",
                (question_hash,),
            ).fetchone()
            if row is None:
                return False
            self.conn.execute(
                "INSERT OR IGNORE INTO cache_vectors (rowid, embedding) VALUES (?, ?)",
                (row[0], serialize_f32(embedding)),
            )
        return True

    def search(self, embedding: list[float], threshold: float) -> CacheMatch | None:
        rows = self.conn.execute(
            """
            SELECT
                e.question_hash,
                e.question,
                e.answer,
                e.thought,
                v.distance
            FROM cache_vectors v
            JOIN cache_entries e ON e.id = v.rowid
            WHERE v.embedding MATCH ?
              AND k = 1
            ORDER BY v.distance
            """,
            (serialize_f32(embedding),),
        ).fetchall()

        if not rows:
            return None

        question_hash, question, answer, thought, distance = rows[0]
        similarity = 1.0 - float(distance)
        if similarity < threshold:
            return None

        return CacheMatch(
            question_hash=question_hash,
            question=question,
            answer=answer,
            thought=thought,
            similarity=similarity,
        )