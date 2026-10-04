"""コマンドライン: ingest / ask / eval / serve。"""
from __future__ import annotations

import argparse
import sys

from . import evaluate as ev
from .config import EVAL_DIR, GEN_MODEL
from .embed import QuotaError
from .service import QAService, run_ingest


def _need_key(dry_run: bool) -> None:
    from .config import get_api_key

    if not dry_run and not get_api_key():
        sys.exit("GEMINI_API_KEY が見つかりません。--dry-run で試すか、.env または環境変数にキーを設定してください。")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragdemo", description="社内文書QA（RAG）デモ")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--dry-run", action="store_true", help="外部APIを使わず、擬似ベクトル＋抜粋回答で動かす")

    a = sub.add_parser("ingest", help="docs/ を取り込む（変更のあったチャンクだけ再埋め込み）")
    add_common(a)
    a = sub.add_parser("ask", help="質問する")
    add_common(a)
    a.add_argument("question")
    a.add_argument("--show-chunks", action="store_true", help="検索で取れたチャンクも表示")
    a = sub.add_parser("eval", help="eval/questions.jsonl で評価する")
    add_common(a)
    a.add_argument("--model", default=GEN_MODEL)
    a.add_argument("--limit", type=int, default=None, help="この回数だけ新規にAPIを呼んで止める")
    a.add_argument("--no-report", action="store_true", help="eval/results_<日付>.md を書かない")
    a = sub.add_parser("serve", help="Web UIを起動する")
    add_common(a)
    a.add_argument("--host", default="127.0.0.1")
    a.add_argument("--port", type=int, default=8000)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _need_key(args.dry_run)
    try:
        if args.cmd == "ingest":
            print(run_ingest(args.dry_run))
        elif args.cmd == "ask":
            svc = QAService(args.dry_run)
            ans = svc.ask(args.question)
            if args.show_chunks:
                for h in ans.hits:
                    c = h.chunk
                    print(f"  - {c.file}「{c.heading}」 rrf={h.score:.4f} vec={h.vec_score:.3f} bm25={h.bm25_score:.2f}")
                print()
            print(ans.text)
        elif args.cmd == "eval":
            svc = QAService(args.dry_run, model=args.model)
            qs = ev.load_questions(ev.default_questions())
            rows, stopped = ev.run_eval(svc, qs, ev.default_cache(args.dry_run), args.limit)
            m = ev.compute_metrics(rows, len(qs))
            mode = "dry-run（擬似）" if args.dry_run else "Gemini実API"
            report = ev.render_report(rows, m, mode, svc.answerer.name, stopped)
            print(report)
            if not args.no_report:
                suffix = "_dryrun" if args.dry_run else ""
                out = EVAL_DIR / f"results_{ev.report_date(rows)}{suffix}.md"
                out.write_text(report, encoding="utf-8")
                print(f"\n保存: {out}")
            if stopped:
                print(f"\n注意: {stopped}。後日 `eval` を再実行すると続きから進みます。")
        elif args.cmd == "serve":
            import uvicorn

            from .web import create_app

            uvicorn.run(create_app(args.dry_run), host=args.host, port=args.port)
    except QuotaError as e:
        sys.exit(f"無料枠の上限に達しました: {e}")
    except RuntimeError as e:  # インデックス未作成・キー未設定など、利用者向けメッセージ付きのもの
        sys.exit(str(e))
    return 0
