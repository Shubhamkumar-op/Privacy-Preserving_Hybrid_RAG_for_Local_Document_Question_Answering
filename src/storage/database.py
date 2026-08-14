import sqlite3
from pathlib import Path


class ChunkDatabase:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_name TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk TEXT NOT NULL
                )
            """)

    def add(self, pdf_name: str, page_number: int, chunk: str):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chunks (pdf_name, page_number, chunk) VALUES (?, ?, ?)",
                (pdf_name, page_number, chunk),
            )

    def search(self, term: str, limit: int = 10):
        with self._connect() as conn:
            return conn.execute(
                "SELECT pdf_name, page_number, chunk FROM chunks WHERE chunk LIKE ? LIMIT ?",
                (f"%{term}%", limit),
            ).fetchall()
