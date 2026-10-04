"""SQLiteにチャンクと埋め込みを保存する。"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY, file TEXT, heading TEXT, text TEXT, doc_title TEXT,
  hash TEXT, vec BLOB, ord INTEGER
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass
class StoredChunk:
    id: str
    file: str
    heading: str
    text: str
    doc_title: str
    hash: str
    vec: np.ndarray


class Store:
    def __init__(self, path: Path) -> None:
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def get_meta(self, key: str) -> str | None:
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

    def hash_to_vec(self) -> dict[str, bytes]:
        return {h: v for h, v in self.db.execute("SELECT hash, vec FROM chunks")}

    def replace_all(self, rows: list[tuple]) -> None:
        """rows: (id, file, heading, text, doc_title, hash, vec_bytes, ord)。1トランザクションで入れ替える。"""
        with self.db:
            self.db.execute("DELETE FROM chunks")
            self.db.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)", rows)

    def load(self) -> list[StoredChunk]:
        out = []
        for id_, f, h, t, dt, hs, vec, _ in self.db.execute("SELECT * FROM chunks ORDER BY ord"):
            out.append(StoredChunk(id_, f, h, t, dt, hs, np.frombuffer(vec, dtype=np.float32)))
        return out

    def fingerprint(self) -> str:
        return hashlib.sha256("".join(c.hash for c in self.load()).encode()).hexdigest()[:16]

    def close(self) -> None:
        self.db.commit()
        self.db.close()
