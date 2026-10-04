"""ハイブリッド検索: ベクトル（コサイン）とBM25を RRF（逆順位融合）で合成する。"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .embed import Embedder, char_ngrams
from .store import StoredChunk

_SEP = r"\sA-Za-z0-9_\-,.、。（）()「」・：:／/"


def tokenize(text: str) -> list[str]:
    """英数字は単語、日本語は文字バイグラム（形態素解析器なしで動かすため）。"""
    toks: list[str] = []
    for m in re.finditer(rf"[A-Za-z0-9_\-,.]+|[^{_SEP}]+", text.lower()):
        s = m.group(0)
        if re.match(r"[a-z0-9]", s):
            toks.append(s.strip(",."))
        elif len(s) == 1:
            toks.append(s)
        else:
            toks.extend(char_ngrams(s, 2))
    return [t for t in toks if t]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 1.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


@dataclass
class Hit:
    chunk: StoredChunk
    score: float  # RRFスコア
    vec_score: float
    bm25_score: float
    coverage: float  # クエリ語のうち（IDF重み付きで）このチャンクに現れる割合


class HybridSearcher:
    def __init__(self, chunks: list[StoredChunk], embedder: Embedder) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.mat = np.stack([c.vec for c in chunks]) if chunks else np.zeros((0, 1), np.float32)
        self.tokens = [tokenize(f"{c.heading} {c.text}") for c in chunks]
        self.tokset = [set(t) for t in self.tokens]
        self.bm25 = BM25(self.tokens)

    def search(self, query: str, k: int = 5, qvec: np.ndarray | None = None, rrf_k: int = 60) -> list[Hit]:
        if not self.chunks:
            return []
        qv = qvec if qvec is not None else self.embedder.embed_query(query)
        vs = self.mat @ qv
        qt = tokenize(query)
        bs = self.bm25.scores(qt)
        rank: dict[int, float] = {}
        for scores in (vs, bs):
            for r, i in enumerate(np.argsort(-scores)):
                rank[int(i)] = rank.get(int(i), 0.0) + 1.0 / (rrf_k + r + 1)
        qset = set(qt)
        total = sum(self.bm25.idf.get(t, 0.0) for t in qset) or 1.0
        top = sorted(rank, key=lambda i: -rank[i])[:k]
        hits = []
        for i in top:
            cov = sum(self.bm25.idf.get(t, 0.0) for t in qset if t in self.tokset[i]) / total
            hits.append(Hit(self.chunks[i], rank[i], float(vs[i]), float(bs[i]), cov))
        return hits
