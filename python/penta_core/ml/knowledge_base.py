"""
Knowledge Base - ingest external sources and build a local reference store.

Provides:
- Markdown ingestion (local files/directories)
- Basic web ingestion (HTML stripped to text)
- SQLite-backed storage for search and retrieval
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Dict, List, Optional
import html.parser
import json
import re
import sqlite3
import urllib.request


@dataclass(frozen=True)
class KnowledgeDocument:
    """A normalized knowledge document."""
    doc_id: str
    title: str
    source: str
    content: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


class _HTMLStripper(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self._chunks.append(data)

    def get_text(self) -> str:
        return " ".join(chunk.strip() for chunk in self._chunks if chunk.strip())


class KnowledgeBase:
    """SQLite-backed knowledge base for ML baseline ingestion."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.storage_path)
        self._init_schema()

    def _init_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT,
                source TEXT,
                content TEXT,
                tags TEXT,
                metadata TEXT
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source)")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def add_document(self, document: KnowledgeDocument) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO documents
            (doc_id, title, source, content, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                document.doc_id,
                document.title,
                document.source,
                document.content,
                ",".join(document.tags),
                json.dumps(document.metadata),
            ),
        )
        self._conn.commit()

    def add_documents(self, documents: Iterable[KnowledgeDocument]) -> None:
        for document in documents:
            self.add_document(document)

    def ingest_markdown(self, path: Path, *, source_label: Optional[str] = None) -> List[KnowledgeDocument]:
        path = Path(path)
        documents: List[KnowledgeDocument] = []

        if path.is_dir():
            for markdown_file in sorted(path.rglob("*.md")):
                documents.extend(self.ingest_markdown(markdown_file, source_label=source_label))
            return documents

        content = path.read_text(encoding="utf-8")
        title = _extract_title(content) or path.stem
        doc_id = f"md:{path.resolve()}"
        document = KnowledgeDocument(
            doc_id=doc_id,
            title=title,
            source=source_label or str(path),
            content=content,
            tags=["markdown"],
            metadata={"path": str(path)},
        )
        self.add_document(document)
        documents.append(document)
        return documents

    def ingest_web(self, url: str, *, title: Optional[str] = None, source_label: Optional[str] = None) -> KnowledgeDocument:
        with urllib.request.urlopen(url) as response:
            raw_html = response.read().decode("utf-8", errors="ignore")
        text = _strip_html(raw_html)
        doc_title = title or _extract_title(raw_html) or url
        document = KnowledgeDocument(
            doc_id=f"web:{url}",
            title=doc_title,
            source=source_label or url,
            content=text,
            tags=["web"],
            metadata={"url": url},
        )
        self.add_document(document)
        return document

    def search(self, query: str, *, limit: int = 10) -> List[KnowledgeDocument]:
        cursor = self._conn.cursor()
        like_query = f"%{query}%"
        cursor.execute(
            """
            SELECT doc_id, title, source, content, tags, metadata
            FROM documents
            WHERE title LIKE ? OR content LIKE ? OR source LIKE ?
            LIMIT ?
            """,
            (like_query, like_query, like_query, limit),
        )
        rows = cursor.fetchall()
        return [
            KnowledgeDocument(
                doc_id=row[0],
                title=row[1],
                source=row[2],
                content=row[3],
                tags=row[4].split(",") if row[4] else [],
                metadata=json.loads(row[5]) if row[5] else {},
            )
            for row in rows
        ]


def connect_knowledge_base(storage_path: Path) -> KnowledgeBase:
    """Create a knowledge base connection."""
    return KnowledgeBase(storage_path)


def _strip_html(raw_html: str) -> str:
    parser = _HTMLStripper()
    parser.feed(raw_html)
    return parser.get_text()


def _extract_title(raw_text: str) -> Optional[str]:
    match = re.search(r"^#\s+(.+)$", raw_text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None
