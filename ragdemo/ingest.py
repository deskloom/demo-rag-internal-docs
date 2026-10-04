"""取り込み。チャンクのハッシュで差分判定し、変わったものだけ再埋め込みする。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunker import Chunk, chunk_markdown
from .embed import Embedder
from .store import Store


@dataclass
class IngestReport:
    total: int
    embedded: int
    reused: int
    removed: int

    def __str__(self) -> str:
        return (
            f"チャンク {self.total} 件"
            f"（再埋め込み {self.embedded}・再利用 {self.reused}・削除 {self.removed}）"
        )


def collect_chunks(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in sorted(docs_dir.glob("*.md")):
        chunks.extend(chunk_markdown(p.name, p.read_text(encoding="utf-8")))
    return chunks


def ingest(docs_dir: Path, store: Store, embedder: Embedder) -> IngestReport:
    chunks = collect_chunks(docs_dir)
    # 埋め込み器が変わったら全再計算（次元や空間が違うため）
    if store.get_meta("embedder") != embedder.name:
        old: dict[str, bytes] = {}
    else:
        old = store.hash_to_vec()

    new_hashes = {c.hash for c in chunks}
    removed = len([h for h in old if h not in new_hashes])
    # 同一内容のチャンクが複数あっても埋め込みは1回だけ
    uniq: dict[str, Chunk] = {}
    for c in chunks:
        if c.hash not in old:
            uniq.setdefault(c.hash, c)
    vecs = embedder.embed_documents([c.embed_text for c in uniq.values()])
    fresh = {h: vecs[i].tobytes() for i, h in enumerate(uniq)}

    rows, seen = [], {}
    for i, c in enumerate(chunks):
        n = seen.get(c.id, 0)
        seen[c.id] = n + 1
        cid = c.id if n == 0 else f"{c.id}#{n}"
        rows.append((cid, c.file, c.heading, c.text, c.doc_title, c.hash, old.get(c.hash) or fresh[c.hash], i))
    store.replace_all(rows)
    store.set_meta("embedder", embedder.name)
    store.db.commit()
    reused = sum(1 for c in chunks if c.hash in old)
    return IngestReport(len(chunks), len(uniq), reused, removed)
