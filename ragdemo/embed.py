"""埋め込み。MockEmbedder は外部APIなしで動く擬似ベクトル、GeminiEmbedder は実API。"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Protocol

import numpy as np

from .config import EMBED_DIM, EMBED_MODEL, MOCK_DIM, get_api_key


class QuotaError(RuntimeError):
    """無料枠の上限（429）に当たった。"""


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


def char_ngrams(text: str, n: int = 2) -> list[str]:
    text = re.sub(r"\s+", "", text.lower())
    return [text[i : i + n] for i in range(max(len(text) - n + 1, 1))]


class MockEmbedder:
    """文字バイグラムをハッシュして符号付きで次元に散らすだけの擬似ベクトル（意味は捉えない）。"""

    name = f"mock-hash-{MOCK_DIM}"

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(MOCK_DIM, dtype=np.float32)
        for g in char_ngrams(text):
            h = int.from_bytes(hashlib.md5(g.encode("utf-8")).digest()[:8], "big")
            v[h % MOCK_DIM] += 1.0 if (h >> 40) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, MOCK_DIM), np.float32)
        return _normalize(np.stack([self._one(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def call_with_retry(fn, tries: int = 3, wait: float = 20.0):
    """429/503 を数回だけ再試行。それでも429なら QuotaError。"""
    from google.genai import errors

    for i in range(tries):
        try:
            return fn()
        except errors.APIError as e:
            code = getattr(e, "code", None)
            if code in (429, 503) and i < tries - 1:
                time.sleep(wait)
                continue
            if code == 429:
                raise QuotaError(str(e)[:300]) from e
            raise


class GeminiEmbedder:
    name = f"{EMBED_MODEL}-{EMBED_DIM}"

    def __init__(self) -> None:
        from google import genai

        key = get_api_key()
        if not key:
            raise RuntimeError("GEMINI_API_KEY がありません。--dry-run で試すか .env を用意してください。")
        self.client = genai.Client(api_key=key)

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(texts), 50):
            batch = texts[i : i + 50]
            resp = call_with_retry(
                lambda b=batch: self.client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=b,
                    config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM),
                )
            )
            out.extend(e.values for e in resp.embeddings)
        return _normalize(np.array(out, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBED_DIM), np.float32)
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def get_embedder(dry_run: bool) -> Embedder:
    return MockEmbedder() if dry_run else GeminiEmbedder()
