"""1ページのWeb UI（FastAPI）。"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .embed import QuotaError
from .service import QAService, run_ingest

PAGE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>社内文書QA</title>
<style>
body{font-family:system-ui,"Yu Gothic",sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#222}
h1{font-size:1.3rem} form{display:flex;gap:.5rem} input{flex:1;padding:.6rem;font-size:1rem}
button{padding:.6rem 1.2rem;font-size:1rem} #ans{white-space:pre-wrap;background:#f5f7fa;border-radius:6px;padding:1rem;margin-top:1rem;min-height:3rem}
.badge{font-size:.8rem;color:#666} details{margin-top:1rem;font-size:.9rem} pre{white-space:pre-wrap}
</style></head><body>
<h1>社内文書QA <span class="badge">__MODE__</span></h1>
<form id="f"><input id="q" placeholder="例: 年次有給休暇はいつまでに申請しますか？" required><button>質問する</button></form>
<div id="ans">質問を入力してください。回答には必ず出典（ファイル名と見出し）が付きます。</div>
<details><summary>検索で取れた文書（上位）</summary><pre id="hits"></pre></details>
<script>
document.getElementById('f').addEventListener('submit', async e => {
  e.preventDefault();
  const a = document.getElementById('ans'); a.textContent = '検索中...';
  let r, j;
  try {
    r = await fetch('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question: document.getElementById('q').value})});
    j = await r.json();
  } catch (err) { a.textContent = 'エラー: サーバーから正常な応答を得られませんでした（' + (r ? r.status : '接続失敗') + '）'; return; }
  if (!r.ok) { a.textContent = 'エラー: ' + (j.detail || r.status); return; }
  a.textContent = j.text;
  document.getElementById('hits').textContent = j.hits.map(h => `${h.file}「${h.heading}」 score=${h.score.toFixed(4)}`).join('\\n');
});
</script></body></html>"""


class AskBody(BaseModel):
    question: str


def create_app(dry_run: bool, ingest_on_start: bool = True, db_path=None, docs_dir=None) -> FastAPI:
    if ingest_on_start:
        kw = {}
        if docs_dir is not None:
            kw["docs_dir"] = docs_dir
        run_ingest(dry_run, db_path=db_path, **kw)  # 差分取り込み（変更が無ければ再埋め込みは0件）
    svc = QAService(dry_run, db_path=db_path)
    app = FastAPI(title="社内文書QA")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE.replace("__MODE__", "dry-run（擬似回答）" if dry_run else svc.answerer.name)

    @app.post("/api/ask")
    def ask(body: AskBody) -> dict:
        q = body.question.strip()
        if not q:
            raise HTTPException(400, "質問が空です")
        try:
            a = svc.ask(q)
        except QuotaError as e:
            raise HTTPException(429, f"無料枠の上限に達しました: {e}")
        except Exception as e:  # APIの一時障害など。内部詳細は出さず502で返す
            raise HTTPException(502, f"回答の生成に失敗しました（{type(e).__name__}）")
        return {
            "answerable": a.answerable,
            "text": a.text,
            "sources": a.sources,
            "hits": [{"file": h.chunk.file, "heading": h.chunk.heading, "score": h.score} for h in a.hits],
        }

    return app
