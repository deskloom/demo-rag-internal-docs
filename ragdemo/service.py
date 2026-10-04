"""取り込み・検索・回答をつなぐ薄い層。CLI・Web・評価が共通で使う。"""
from __future__ import annotations

from pathlib import Path

from .answer import Answer, get_answerer
from .config import DOCS_DIR, GEN_MODEL, TOP_K, index_path
from .embed import get_embedder
from .ingest import IngestReport, ingest
from .search import HybridSearcher
from .store import Store


def run_ingest(dry_run: bool, docs_dir: Path = DOCS_DIR, db_path: Path | None = None) -> IngestReport:
    store = Store(db_path or index_path(dry_run))
    try:
        return ingest(docs_dir, store, get_embedder(dry_run))
    finally:
        store.close()


class QAService:
    def __init__(self, dry_run: bool, db_path: Path | None = None, model: str = GEN_MODEL) -> None:
        self.dry_run = dry_run
        self.embedder = get_embedder(dry_run)
        self.answerer = get_answerer(dry_run, model)
        store = Store(db_path or index_path(dry_run))
        self.chunks = store.load()
        self.fingerprint = store.fingerprint()
        store.close()
        if not self.chunks:
            raise RuntimeError("インデックスが空です。先に `ingest` を実行してください。")
        self.searcher = HybridSearcher(self.chunks, self.embedder)

    def ask(self, question: str, qvec=None) -> Answer:
        hits = self.searcher.search(question, k=TOP_K, qvec=qvec)
        return self.answerer.answer(question, hits)
