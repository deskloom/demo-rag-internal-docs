"""評価。出典の一致率・「記載なし」質問の拒否率・回答キーワード含有率・検索再現率を数字で出す。

質問ごとの結果を eval/cache_<mode>.jsonl に追記保存する。無料枠の上限で止まっても、
再実行すると済んだ質問は飛ばして続きから進む（質問文・モデル・埋め込み器・文書の内容・システムプロンプト・TOP_K・RRF_K のどれかが変わると無効）。
出典一致・キーワード判定は保存せず、毎回、保存済みの回答と出典から再計算する。
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .answer import SYSTEM_PROMPT
from .config import EVAL_DIR, GEN_MODEL, TOP_K
from .embed import QuotaError
from .search import RRF_K
from .service import QAService


def load_questions(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def source_matches(expected: dict, file: str, heading: str) -> bool:
    return file == expected["file"] and expected.get("heading", "") in heading


def evaluate_one(svc: QAService, q: dict) -> dict:
    qvec = svc.embedder.embed_query(q["question"])
    ans = svc.ask(q["question"], qvec=qvec)
    return {
        "id": q["id"],
        "question": q["question"],
        "type": q["type"],
        "cfg": run_config(svc),
        "generated": date.today().isoformat(),
        "answerable": ans.answerable,
        "answer": ans.text,
        "cited": ans.sources,
        "retrieved": [{"file": h.chunk.file, "heading": h.chunk.heading} for h in ans.hits],
    }


def run_config(svc: QAService) -> dict:
    """結果を再利用してよい条件。どれか変わったらキャッシュは無効。"""
    return {
        "model": svc.answerer.name,
        "embedder": svc.embedder.name,
        "fingerprint": svc.fingerprint,  # 文書（チャンク）の内容
        "prompt": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12],
        "top_k": TOP_K,
        "rrf_k": RRF_K,
    }


def score_row(row: dict, q: dict) -> dict:
    """保存済みの回答・出典から、期待値との一致を毎回計算し直す（保存値は信用しない）。"""
    row = {k: v for k, v in row.items() if k not in ("cited_hits", "retrieved_hits", "keywords_ok")}
    exp = q.get("expected_sources", [])
    if exp:
        row["cited_hits"] = [any(source_matches(e, c["file"], c["heading"]) for c in row["cited"]) for e in exp]
        row["retrieved_hits"] = [any(source_matches(e, r["file"], r["heading"]) for r in row["retrieved"]) for e in exp]
        must = q.get("must_include", [])
        row["keywords_ok"] = all(m in row["answer"] for m in must) if row["answerable"] else False
    return row


def _key(id_: str, question: str, cfg: dict) -> str:
    return json.dumps([id_, question, cfg], sort_keys=True, ensure_ascii=False)


def run_eval(svc: QAService, questions: list[dict], cache_path: Path, limit: int | None = None) -> tuple[list[dict], str | None]:
    """戻り値: (質問順の結果行。未実行は含まない, 中断理由またはNone)"""
    cache: dict[str, dict] = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "cfg" in r:  # 旧形式の行は条件が不明なので使わない
                    cache[_key(r["id"], r["question"], r["cfg"])] = r
    cfg = run_config(svc)
    rows: list[dict] = []
    stopped: str | None = None
    fresh_calls = 0
    for q in questions:
        k = _key(q["id"], q["question"], cfg)
        if k in cache:
            rows.append(score_row(cache[k], q))
            continue
        if limit is not None and fresh_calls >= limit:
            stopped = f"--limit {limit} に達したため中断"
            break
        try:
            row = evaluate_one(svc, q)
        except QuotaError as e:
            stopped = f"無料枠の上限（429）で中断: {e}"
            break
        fresh_calls += 1
        with cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(score_row(row, q))
    return rows, stopped


@dataclass
class Metrics:
    n_run: int
    n_total: int
    n_answerable: int
    n_none: int
    source_all: int  # 期待出典をすべて引用できた質問数
    source_any: int  # 期待出典を1つ以上引用できた質問数
    keywords_ok: int
    retrieval_all: int  # 検索上位5件に期待出典がすべて入った質問数
    false_refusal: int  # 答えがあるのに「記載なし」とした数
    refused_ok: int  # 答えが無い質問で正しく拒否した数
    n_answerable_run: int
    n_none_run: int


def compute_metrics(rows: list[dict], n_total: int) -> Metrics:
    ans = [r for r in rows if r["type"] != "none"]
    non = [r for r in rows if r["type"] == "none"]
    return Metrics(
        n_run=len(rows),
        n_total=n_total,
        n_answerable=len(ans),
        n_none=len(non),
        source_all=sum(all(r["cited_hits"]) for r in ans),
        source_any=sum(any(r["cited_hits"]) for r in ans),
        keywords_ok=sum(bool(r["keywords_ok"]) for r in ans),
        retrieval_all=sum(all(r["retrieved_hits"]) for r in ans),
        false_refusal=sum(not r["answerable"] for r in ans),
        refused_ok=sum(not r["answerable"] for r in non),
        n_answerable_run=len(ans),
        n_none_run=len(non),
    )


def _pct(a: int, b: int) -> str:
    return f"{a}/{b}（{a / b * 100:.0f}%）" if b else "対象なし"


def render_report(rows: list[dict], m: Metrics, mode: str, model: str, stopped: str | None) -> str:
    gen = sorted({r["generated"] for r in rows})
    gen_txt = "・".join(gen) if gen else "未実行"
    L = [f"# 評価結果（{mode}）", "", f"- 結果の生成日: {gen_txt}"]
    L.append(f"- 生成モデル: `{model}`（dry-run時は抜粋を返すだけの擬似回答）")
    L.append(f"- 実行できた質問: **{m.n_run} / {m.n_total} 問**")
    if stopped:
        L.append(f"- 中断理由: {stopped}。**実行できた分だけを集計**しています。")
    L += ["", "## 指標", "", "| 指標 | 結果 |", "|---|---|"]
    L.append(f"| 出典の一致率（期待した出典すべてを引用） | {_pct(m.source_all, m.n_answerable)} |")
    L.append(f"| 出典の一致率（期待した出典を1つ以上引用） | {_pct(m.source_any, m.n_answerable)} |")
    L.append(f"| 答えが無い質問で正しく拒否した率 | {_pct(m.refused_ok, m.n_none)} |")
    L.append(f"| 答えがあるのに拒否した率（誤拒否） | {_pct(m.false_refusal, m.n_answerable)} |")
    L.append(f"| 回答に必須キーワードが含まれた率（簡易な正確さ） | {_pct(m.keywords_ok, m.n_answerable)} |")
    L.append(f"| 検索上位{TOP_K}件に期待出典がすべて入った率（検索単体） | {_pct(m.retrieval_all, m.n_answerable)} |")
    L += ["", "## 質問ごとの結果", "", "| ID | 種別 | 質問 | 回答可否 | 引用した出典 | 判定 |", "|---|---|---|---|---|---|"]
    for r in rows:
        cited = "<br>".join(f"{c['file']}「{c['heading']}」" for c in r["cited"]) or "-"
        if r["type"] == "none":
            verdict = "OK（拒否）" if not r["answerable"] else "NG（答えてしまった）"
        elif not r["answerable"]:
            verdict = "NG（誤拒否）"
        else:
            parts = [
                "出典" + ("OK" if all(r["cited_hits"]) else "NG"),
                "キーワード" + ("OK" if r["keywords_ok"] else "NG"),
            ]
            verdict = " / ".join(parts)
        L.append(f"| {r['id']} | {r['type']} | {r['question']} | {'回答' if r['answerable'] else '拒否'} | {cited} | {verdict} |")
    L.append("")
    return "\n".join(L)


def report_date(rows: list[dict]) -> str:
    """レポートのファイル名に使う日付。キャッシュ再利用時に今日の日付を名乗らないよう、結果の生成日の最新を使う。"""
    return max((r["generated"] for r in rows), default=date.today().isoformat())


def default_cache(dry_run: bool) -> Path:
    return EVAL_DIR / ("cache_mock.jsonl" if dry_run else "cache_gemini.jsonl")


def default_questions() -> Path:
    return EVAL_DIR / "questions.jsonl"


__all__ = ["report_date", "run_eval", "compute_metrics", "render_report", "load_questions", "default_cache", "default_questions", "GEN_MODEL"]
