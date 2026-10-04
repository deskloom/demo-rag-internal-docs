"""外部APIなしで通るテスト（モック埋め込み・モック/スタブ回答）。"""
import json
import shutil
from pathlib import Path

import pytest

from ragdemo import evaluate as ev
from ragdemo.answer import Answer, resolve_answer
from ragdemo.chunker import chunk_markdown
from ragdemo.config import DOCS_DIR, NO_ANSWER_TEXT
from ragdemo.embed import MockEmbedder
from ragdemo.ingest import ingest
from ragdemo.search import BM25, HybridSearcher, tokenize
from ragdemo.service import QAService, run_ingest
from ragdemo.store import Store


# ---------- chunker ----------
def test_chunk_by_heading_and_title():
    md = "# タイトル\n\n## A\n" + "あ" * 100 + "\n\n## B\n" + "い" * 100 + "\n"
    cs = chunk_markdown("x.md", md)
    assert [c.heading for c in cs] == ["A", "B"]
    assert cs[0].doc_title == "タイトル"


def test_short_sections_are_merged():
    md = "# T\n\n## A\n短い。\n\n## B\n" + "い" * 100 + "\n"
    cs = chunk_markdown("x.md", md)
    assert len(cs) == 1 and cs[0].heading == "A／B"
    assert "短い。" in cs[0].text


def test_trailing_short_section_merges_into_previous():
    md = "# T\n\n## A\n" + "あ" * 100 + "\n\n## B\n短い。\n"
    cs = chunk_markdown("x.md", md)
    assert len(cs) == 1 and cs[0].heading == "A／B"


def test_long_section_is_split_within_limit():
    body = "".join(f"これは{i}番目の文です。" + "あ" * 30 + "。" for i in range(30))
    cs = chunk_markdown("x.md", f"# T\n\n## A\n{body}\n", max_chars=200)
    assert len(cs) > 1
    assert all(len(c.text) <= 200 + 60 for c in cs)  # 1文が長い場合のみ少し超えうる
    assert cs[1].heading == "A（続き1）"
    assert "".join(c.text for c in cs) == body


def test_real_docs_chunk_sanity():
    files = sorted(DOCS_DIR.glob("*.md"))
    assert 8 <= len(files) <= 15
    for p in files:
        cs = chunk_markdown(p.name, p.read_text(encoding="utf-8"))
        assert cs and all(c.heading and c.text for c in cs)


# ---------- ingest（差分） ----------
@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    shutil.copytree(DOCS_DIR, d)
    return d


def test_ingest_is_incremental(docs, tmp_path):
    store = Store(tmp_path / "i.db")
    emb = MockEmbedder()
    r1 = ingest(docs, store, emb)
    assert r1.embedded == r1.total and r1.reused == 0

    r2 = ingest(docs, store, emb)  # 変更なし
    assert r2.embedded == 0 and r2.reused == r2.total

    f = docs / "leave.md"
    f.write_text(f.read_text(encoding="utf-8").replace("夏季休暇は7月から9月の間に3日間", "夏季休暇は7月から9月の間に5日間"), encoding="utf-8")
    r3 = ingest(docs, store, emb)  # 1チャンクだけ変更
    assert r3.embedded == 1 and r3.removed == 1 and r3.reused == r3.total - 1

    (docs / "training.md").unlink()  # 文書の削除
    r4 = ingest(docs, store, emb)
    assert r4.removed >= 1
    assert all(c.file != "training.md" for c in store.load())
    store.close()


def test_embedder_change_triggers_full_reembed(docs, tmp_path):
    store = Store(tmp_path / "i.db")
    ingest(docs, store, MockEmbedder())
    store.set_meta("embedder", "other-model")
    r = ingest(docs, store, MockEmbedder())
    assert r.embedded == r.total
    store.close()


# ---------- search ----------
def test_tokenize_japanese_bigrams_and_ascii():
    assert tokenize("有給休暇") == ["有給", "給休", "休暇"]
    assert "vpn" in tokenize("VPNに接続")


def test_bm25_prefers_matching_doc():
    docs = [tokenize("経費の申請期限は翌月5営業日"), tokenize("夏季休暇は3日間"), tokenize("パスワードは12文字以上")]
    s = BM25(docs).scores(tokenize("夏季休暇は何日"))
    assert int(s.argmax()) == 1


def test_hybrid_search_finds_expected_chunk(docs, tmp_path):
    store = Store(tmp_path / "i.db")
    emb = MockEmbedder()
    ingest(docs, store, emb)
    s = HybridSearcher(store.load(), emb)
    hits = s.search("夏季休暇は何日取れますか", k=3)
    assert any(h.chunk.file == "leave.md" and "第6条" in h.chunk.heading for h in hits)
    hits = s.search("パスワードの有効期限", k=3)
    assert hits[0].chunk.file == "it_faq.md"
    store.close()


# ---------- answer（出典・拒否の機械ルール） ----------
class _H:  # Hitの最小スタブ
    def __init__(self, file, heading):
        self.chunk = type("C", (), {"file": file, "heading": heading, "text": "本文"})()


HITS = [_H("a.md", "第1条"), _H("b.md", "第2条")]


def test_resolve_answer_adds_sources():
    a = resolve_answer({"answerable": True, "answer": "3日です。", "source_ids": [2]}, HITS)
    assert a.answerable and a.sources == [{"file": "b.md", "heading": "第2条"}]
    assert "出典:" in a.text and "b.md「第2条」" in a.text


def test_resolve_answer_refuses_without_valid_sources():
    for parsed in (
        {"answerable": True, "answer": "3日です。", "source_ids": []},
        {"answerable": True, "answer": "3日です。", "source_ids": [9]},
        {"answerable": True, "answer": "3日です。"},
        {"answerable": False, "answer": "", "source_ids": [1]},
        {"answerable": True, "answer": "  ", "source_ids": [1]},
    ):
        a = resolve_answer(parsed, HITS)
        assert not a.answerable and a.text == NO_ANSWER_TEXT and a.sources == []


# ---------- service / eval（モック） ----------
@pytest.fixture
def svc(docs, tmp_path):
    db = tmp_path / "i.db"
    run_ingest(True, docs_dir=docs, db_path=db)
    return QAService(True, db_path=db)


def test_mock_service_answers_with_source_and_refuses(svc):
    a = svc.ask("夏季休暇は何日取れますか？")
    assert a.answerable and a.sources and "出典:" in a.text
    a = svc.ask("zzzzqqqq xxxxwwww")
    assert not a.answerable and a.text == NO_ANSWER_TEXT


def test_eval_runs_and_resumes_from_cache(svc, tmp_path):
    qs = ev.load_questions(ev.default_questions())
    assert 15 <= len(qs) <= 30
    assert {q["type"] for q in qs} == {"single", "multi", "none"}
    cache = tmp_path / "cache.jsonl"
    rows1, stopped = ev.run_eval(svc, qs, cache, limit=5)
    assert len(rows1) == 5 and stopped
    rows2, stopped = ev.run_eval(svc, qs, cache)  # 続きから（5件はキャッシュ）
    assert len(rows2) == len(qs) and stopped is None
    assert len(cache.read_text(encoding="utf-8").splitlines()) == len(qs)
    m = ev.compute_metrics(rows2, len(qs))
    assert m.n_run == len(qs) and 0 <= m.source_all <= m.n_answerable
    assert "出典の一致率" in ev.render_report(rows2, m, "dry-run", "mock", None)


def test_eval_quota_stop_keeps_finished_rows(svc, tmp_path, monkeypatch):
    from ragdemo.embed import QuotaError

    qs = ev.load_questions(ev.default_questions())
    calls = {"n": 0}
    orig = ev.evaluate_one

    def flaky(s, q):
        calls["n"] += 1
        if calls["n"] == 4:
            raise QuotaError("429")
        return orig(s, q)

    monkeypatch.setattr(ev, "evaluate_one", flaky)
    rows, stopped = ev.run_eval(svc, qs, tmp_path / "c.jsonl")
    assert len(rows) == 3 and "429" in stopped


def test_questions_expected_sources_exist_in_docs():
    """期待出典のファイルと見出しキーワードが実際の文書に存在する（評価データの typo 検出）。"""
    from ragdemo.ingest import collect_chunks

    chunks = collect_chunks(DOCS_DIR)
    for q in ev.load_questions(ev.default_questions()):
        for e in q["expected_sources"]:
            assert any(c.file == e["file"] and e["heading"] in c.heading for c in chunks), q["id"]


# ---------- web ----------
def test_web_api(docs, tmp_path):
    from fastapi.testclient import TestClient

    from ragdemo.web import create_app

    app = create_app(True, db_path=tmp_path / "w.db", docs_dir=docs)
    c = TestClient(app)
    assert "社内文書QA" in c.get("/").text
    r = c.post("/api/ask", json={"question": "夏季休暇は何日取れますか？"})
    assert r.status_code == 200 and r.json()["sources"]
    assert c.post("/api/ask", json={"question": "  "}).status_code == 400


def test_api_key_loader_never_needs_repo_env(monkeypatch, tmp_path):
    from ragdemo import config

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    f = tmp_path / "e.env"
    f.write_text('# c\nGEMINI_API_KEY="abc"\n', encoding="utf-8")
    monkeypatch.setenv("GEMINI_ENV_FILE", str(f))
    assert config.get_api_key() == "abc"


# ---------- レビュー指摘の回帰テスト ----------
def test_hash_changes_when_doc_title_changes():  # [A]
    a = chunk_markdown("x.md", "# 旧タイトル\n\n## A\n" + "あ" * 100)[0]
    b = chunk_markdown("x.md", "# 新タイトル\n\n## A\n" + "あ" * 100)[0]
    assert a.text == b.text and a.hash != b.hash


def test_title_change_triggers_reembed(docs, tmp_path):  # [A]
    store = Store(tmp_path / "i.db")
    ingest(docs, store, MockEmbedder())
    f = docs / "travel.md"
    f.write_text(f.read_text(encoding="utf-8").replace("# 出張旅費規程", "# 出張旅費規程（改訂）", 1), encoding="utf-8")
    r = ingest(docs, store, MockEmbedder())
    n_travel = sum(1 for c in store.load() if c.file == "travel.md")
    assert r.embedded == n_travel > 0
    store.close()


def test_preamble_before_first_heading_is_kept():  # [B]
    md = "# 規程\n\n" + "前文です。" * 15 + "\n\n## 第1条\n" + "あ" * 100 + "\n"
    cs = chunk_markdown("x.md", md)
    assert [c.heading for c in cs] == ["規程", "第1条"]
    assert "前文です。" in cs[0].text


def test_doc_without_h2_is_kept_under_title():  # [B]
    cs = chunk_markdown("x.md", "# メモ\n\n" + "見出しのない本文。" * 12 + "\n")
    assert len(cs) == 1 and cs[0].heading == "メモ" and "見出しのない本文。" in cs[0].text


def test_doc_without_any_heading_uses_filename():  # [B]
    cs = chunk_markdown("plain.md", "見出しなしの本文。" * 12)
    assert len(cs) == 1 and cs[0].heading == "plain.md"


def test_bm25_zero_match_chunks_not_ranked_in_rrf(docs, tmp_path):  # [C]
    from ragdemo.search import RRF_K

    store = Store(tmp_path / "i.db")
    emb = MockEmbedder()
    ingest(docs, store, emb)
    s = HybridSearcher(store.load(), emb)
    q = "夏季休暇は何日取れますか"
    qv = emb.embed_query(q)
    vs = s.mat @ qv
    bs = s.bm25.scores(tokenize(q))
    assert (bs == 0).any() and (bs > 0).any()
    zero = int(next(i for i in range(len(bs)) if bs[i] == 0))
    vrank = int((vs > vs[zero]).sum())  # ベクトル順位（0始まり）
    top = s.search(q, k=len(s.chunks))
    got = next(h.score for h in top if h.chunk.id == s.chunks[zero].id)
    assert got == pytest.approx(1.0 / (RRF_K + vrank + 1))  # BM25側の加算が無い
    store.close()


def test_eval_cache_invalidated_by_config_and_rescored(svc, tmp_path, monkeypatch):  # [D]
    qs = ev.load_questions(ev.default_questions())[:3]
    cache = tmp_path / "c.jsonl"
    ev.run_eval(svc, qs, cache)
    n = len(cache.read_text(encoding="utf-8").splitlines())
    ev.run_eval(svc, qs, cache)
    assert len(cache.read_text(encoding="utf-8").splitlines()) == n  # 再利用
    for patch in ("ragdemo.evaluate.TOP_K", "ragdemo.evaluate.RRF_K"):
        monkeypatch.setattr(patch, 99)
        ev.run_eval(svc, qs, cache)
        monkeypatch.undo()
    assert len(cache.read_text(encoding="utf-8").splitlines()) == n + 2 * len(qs)
    monkeypatch.setattr("ragdemo.evaluate.SYSTEM_PROMPT", "別のプロンプト")
    ev.run_eval(svc, qs, cache)
    assert len(cache.read_text(encoding="utf-8").splitlines()) == n + 3 * len(qs)


def test_eval_scores_recomputed_not_trusted(svc, tmp_path):  # [D]
    qs = ev.load_questions(ev.default_questions())[:1]
    cache = tmp_path / "c.jsonl"
    ev.run_eval(svc, qs, cache)
    r = json.loads(cache.read_text(encoding="utf-8"))
    assert "cited_hits" not in r and "keywords_ok" not in r and r["generated"]
    r["cited_hits"], r["keywords_ok"] = [True], True  # 改ざんしても無視される
    r["answerable"], r["cited"], r["answer"] = False, [], "文書に記載がありません。"
    cache.write_text(json.dumps(r, ensure_ascii=False) + "\n", encoding="utf-8")
    rows, _ = ev.run_eval(svc, qs, cache)
    assert rows[0]["cited_hits"] == [False] and rows[0]["keywords_ok"] is False
    assert ev.report_date(rows) == r["generated"]


def test_web_returns_502_on_unexpected_error(docs, tmp_path, monkeypatch):  # [F]
    from fastapi.testclient import TestClient

    from ragdemo import service
    from ragdemo.web import create_app

    app = create_app(True, db_path=tmp_path / "w.db", docs_dir=docs)
    monkeypatch.setattr(service.QAService, "ask", lambda self, q, qvec=None: (_ for _ in ()).throw(ValueError("boom")))
    r = TestClient(app).post("/api/ask", json={"question": "夏季休暇は？"})
    assert r.status_code == 502 and "boom" not in r.text


def test_cli_ask_before_ingest_exits_with_message(tmp_path, monkeypatch):  # [G]
    from ragdemo import cli, config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    with pytest.raises(SystemExit) as e:
        cli.main(["ask", "夏季休暇は？", "--dry-run"])
    assert "ingest" in str(e.value)
