"""Markdownを見出し単位でチャンク分割する。長い節は分割、短い節は結合する。"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .config import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS


@dataclass
class Chunk:
    file: str
    heading: str
    text: str  # 本文（見出しを含まない）
    doc_title: str

    @property
    def id(self) -> str:  # ingest側で重複時に連番を付けて一意にする
        return f"{self.file}::{self.heading}"

    @property
    def embed_text(self) -> str:
        return f"{self.doc_title} / {self.heading}\n{self.text}"

    @property
    def hash(self) -> str:
        return hashlib.sha256(
            f"{self.file}\n{self.heading}\n{self.text}".encode("utf-8")
        ).hexdigest()


def _split_sections(md: str) -> tuple[str, list[tuple[str, str]]]:
    title = ""
    sections: list[tuple[str, str]] = []
    cur_head: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if cur_head is not None:
            body = "\n".join(buf).strip()
            if body:
                sections.append((cur_head, body))

    for line in md.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if m:
            level, head = len(m.group(1)), m.group(2)
            if level == 1 and not title:
                title = head
                continue
            flush()
            cur_head, buf = head, []
        else:
            buf.append(line)
    flush()
    return title, sections


def _split_long(text: str, max_chars: int) -> list[str]:
    """段落→文（。）の順に、max_chars以内へ貪欲に詰める。"""
    units: list[str] = []
    for para in re.split(r"\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
        else:
            units.extend(s for s in re.split(r"(?<=。)", para) if s.strip())
    parts: list[str] = []
    cur = ""
    for u in units:
        if cur and len(cur) + len(u) > max_chars:
            parts.append(cur)
            cur = u
        else:
            cur = cur + u if cur else u
    if cur:
        parts.append(cur)
    return parts


def chunk_markdown(
    file: str,
    md: str,
    max_chars: int = MAX_CHUNK_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[Chunk]:
    title, sections = _split_sections(md)
    title = title or file

    # 1) 短い節は次の節へ結合（最後の短い節は直前へ結合）
    merged: list[tuple[str, str]] = []
    carry: tuple[str, str] | None = None
    for head, body in sections:
        if carry:
            head, body = f"{carry[0]}／{head}", f"{carry[1]}\n{body}"
            carry = None
        if len(body) < min_chars:
            carry = (head, body)
        else:
            merged.append((head, body))
    if carry:
        if merged:
            ph, pb = merged[-1]
            merged[-1] = (f"{ph}／{carry[0]}", f"{pb}\n{carry[1]}")
        else:
            merged.append(carry)

    # 2) 長い節は分割（見出しに「（続きn）」を付ける）
    chunks: list[Chunk] = []
    for head, body in merged:
        if len(body) <= max_chars:
            chunks.append(Chunk(file, head, body, title))
            continue
        for i, part in enumerate(_split_long(body, max_chars)):
            h = head if i == 0 else f"{head}（続き{i}）"
            chunks.append(Chunk(file, h, part, title))
    return chunks
