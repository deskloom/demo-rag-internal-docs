"""設定とAPIキーの読み込み。キーの値は表示・保存しない。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "eval"

GEN_MODEL = os.environ.get("RAG_GEN_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768  # gemini-embedding-001 の出力次元を768に絞る（正規化は自前で行う）
MOCK_DIM = 384

TOP_K = 5
MAX_CHUNK_CHARS = 500
MIN_CHUNK_CHARS = 40

NO_ANSWER_TEXT = "文書に記載がありません。"


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def get_api_key() -> str | None:
    """環境変数 GEMINI_API_KEY、無ければ GEMINI_ENV_FILE（既定: リポジトリ直下の .env）から読む。"""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    env_file = Path(os.environ.get("GEMINI_ENV_FILE", ROOT / ".env"))
    return _parse_env_file(env_file).get("GEMINI_API_KEY") or None


def index_path(dry_run: bool) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR / ("index_mock.db" if dry_run else "index_gemini.db")
