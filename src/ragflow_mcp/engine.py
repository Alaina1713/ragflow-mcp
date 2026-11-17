from __future__ import annotations
import hashlib, re, sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Protocol
import numpy as np

class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...

class SentenceTransformerEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name, self._model = model_name, None
    def encode(self, texts):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)

@dataclass
class SearchResult:
    document_id: str
    title: str
    source: str
    chunk_index: int
    text: str
    score: float

def chunk_text(text: str, chunk_size=900, overlap=150) -> list[str]:
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size.")
    text = re.sub(r"\\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start: end = boundary
        part = text[start:end].strip()
        if part: chunks.append(part)
        if end >= len(text): break
        start = max(end - overlap, start + 1)
    return chunks

class KnowledgeBase:
    def __init__(self, db_path="data/ragflow.sqlite3", embedder=None, chunk_size=900, overlap=150):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.chunk_size, self.overlap = chunk_size, overlap
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS documents(
                document_id TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
            db.execute("""CREATE TABLE IF NOT EXISTS chunks(
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL, text TEXT NOT NULL, embedding BLOB NOT NULL,
                dimensions INTEGER NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")

    def _connect(self):
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        return db

    def add_text(self, title, text, source="manual"):
        if not title.strip() or not text.strip(): raise ValueError("Both title and text are required.")
        digest = hashlib.sha256(text.encode()).hexdigest()
        with self._connect() as db:
            old = db.execute("SELECT document_id FROM documents WHERE content_hash=?", (digest,)).fetchone()
            if old: return {"document_id": old["document_id"], "chunks": 0, "duplicate": True}
        parts = chunk_text(text, self.chunk_size, self.overlap)
        vectors = self.embedder.encode(parts)
        if len(vectors) != len(parts): raise RuntimeError("Unexpected embedding count.")
        doc_id = digest[:16]
        with self._connect() as db:
            db.execute("INSERT INTO documents(document_id,title,source,content_hash) VALUES(?,?,?,?)",
                       (doc_id, title.strip(), source, digest))
            for i, (part, vector) in enumerate(zip(parts, vectors)):
                vector = np.asarray(vector, dtype=np.float32).reshape(-1)
                db.execute("INSERT INTO chunks(document_id,chunk_index,text,embedding,dimensions) VALUES(?,?,?,?,?)",
                           (doc_id, i, part, vector.tobytes(), len(vector)))
        return {"document_id": doc_id, "chunks": len(parts), "duplicate": False}

    def ingest_file(self, path):
        path = Path(path).expanduser().resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        if path.suffix.lower() not in {".txt", ".md"}: raise ValueError("Only .txt and .md are supported.")
        return self.add_text(path.name, path.read_text(encoding="utf-8"), str(path))

    def search(self, query, top_k=5):
        if not query.strip(): raise ValueError("Query cannot be empty.")
        if not 1 <= top_k <= 20: raise ValueError("top_k must be between 1 and 20.")
        q = np.asarray(self.embedder.encode([query])[0], dtype=np.float32).reshape(-1)
        with self._connect() as db:
            rows = db.execute("""SELECT c.*,d.title,d.source FROM chunks c
                JOIN documents d ON d.document_id=c.document_id""").fetchall()
        found = []
        for r in rows:
            v = np.frombuffer(r["embedding"], dtype=np.float32)
            if len(v) != len(q): continue
            denom = float(np.linalg.norm(v) * np.linalg.norm(q))
            score = float(np.dot(v, q) / denom) if denom else 0.0
            found.append(SearchResult(r["document_id"], r["title"], r["source"],
                                      r["chunk_index"], r["text"], score))
        return sorted(found, key=lambda x: x.score, reverse=True)[:top_k]

    def list_documents(self):
        with self._connect() as db:
            rows = db.execute("""SELECT d.document_id,d.title,d.source,d.created_at,
                COUNT(c.chunk_id) chunks FROM documents d LEFT JOIN chunks c
                ON c.document_id=d.document_id GROUP BY d.document_id ORDER BY d.created_at DESC""").fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, document_id):
        with self._connect() as db:
            db.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            cur = db.execute("DELETE FROM documents WHERE document_id=?", (document_id,))
            return cur.rowcount > 0
