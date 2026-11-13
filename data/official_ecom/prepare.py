"""离线校验并导出数据；--index 显式调用 embedding API，写独立索引。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare():
    docs = json.loads((ROOT / "seed.json").read_text(encoding="utf-8"))
    docs += json.loads((ROOT / "extra_seed.json").read_text(encoding="utf-8"))
    challenges = json.loads((ROOT / "challenges.json").read_text(encoding="utf-8"))
    chunks, cases, manifest = [], [], []
    # 文档级划分，同一条规则的两个问法保持在同一集合；不是独立的两条知识。
    dev_docs = {"pdd_help", "jd_coupon", "jd_invoice"}
    for doc in docs:
        assert doc["platform"] in {"pdd", "jd"}
        assert doc["source_url"].startswith("https://")
        sections = []
        for key, title, content, formal, colloquial in doc["rules"]:
            chunk_id = doc["doc_id"] + "." + key
            chunk = dict(chunk_id=chunk_id, doc_id=doc["doc_id"], platform=doc["platform"],
                         title=doc["title"], section=title, content=content,
                         source_url=doc["source_url"], effective_date=doc["effective_date"],
                         checked_at="2026-10-29", access=doc["access"],
                         content_kind="source_based_paraphrase")
            chunks.append(chunk)
            sections.append(f"## {title}\n\n片段ID：{chunk_id}\n\n{content}\n")
            for number, query in enumerate((formal, colloquial), 1):
                cases.append(dict(id=f"{chunk_id}.q{number}", query=query,
                                  platform=doc["platform"], relevant_chunk_ids=[chunk_id],
                                  reference_answer=content, family_id=chunk_id,
                                  type="direct" if number == 1 else "paraphrase_or_boundary",
                                  split="dev" if doc["doc_id"] in dev_docs else "test"))
        md_path = ROOT / "knowledge" / doc["platform"] / (doc["doc_id"] + ".md")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(f"# {doc['title']}\n\n来源：{doc['source_url']}\n\n"
                           "这是有来源的摘要，不是官方全文；检索核验日期：2026-10-29。\n\n"
                           + "\n".join(sections), encoding="utf-8")
        manifest.append({k: v for k, v in doc.items() if k != "rules"} |
                        {"checked_at": "2026-10-29", "chunk_count": len(doc["rules"])})
    by_id = {c["chunk_id"]: c for c in chunks}
    assert len(by_id) == len(chunks), "duplicate chunk ids"
    for case in challenges:
        case["family_id"] = case["id"]
        cases.append(case)
    assert len({c["id"] for c in cases}) == len(cases), "duplicate case ids"
    assert len({c["query"] for c in cases}) == len(cases), "duplicate queries"
    assert len({c["content"] for c in chunks}) == len(chunks), "duplicate content"
    for case in cases:
        for cid in case["relevant_chunk_ids"]:
            assert cid in by_id, f"missing evidence: {cid}"
            assert by_id[cid]["platform"] == case["platform"], "cross-platform evidence"
        evidence_splits = {"dev" if by_id[cid]["doc_id"] in dev_docs else "test"
                           for cid in case["relevant_chunk_ids"]}
        if len(evidence_splits) > 1:
            case["split"] = "diagnostic"  # 混合证据仅作诊断，不放入调参/最终测试。
        elif evidence_splits:
            case["split"] = next(iter(evidence_splits))
    for platform in ("pdd", "jd"):
        selected = [c for c in chunks if c["platform"] == platform]
        questions = [c for c in cases if c["platform"] == platform]
        out = ROOT / "exports" / platform
        write_json(out / "chunks.json", selected)
        write_json(out / "cases_all.json", questions)
        for split in ("dev", "test"):
            # 现有评估器只支持有证据问答；其他题单独留作行为测试，不能伪装为空召回失败。
            answerable = [c for c in questions if c["split"] == split and c["type"] in
                          {"direct", "paraphrase_or_boundary", "multi_evidence"}]
            write_json(out / f"rag_{split}.json", [
                {k: c[k] for k in ("id", "query", "relevant_chunk_ids", "reference_answer")}
                for c in answerable])
        write_json(out / "behavior_cases.json", [c for c in questions if c["type"] not in
                   {"direct", "paraphrase_or_boundary", "multi_evidence"}])
    write_json(ROOT / "sources.json", manifest)
    stats = {p: {"documents": sum(d["platform"] == p for d in docs),
                 "chunks": sum(c["platform"] == p for c in chunks),
                 "cases": sum(c["platform"] == p for c in cases)} for p in ("pdd", "jd")}
    write_json(ROOT / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return chunks


def index(platform, chunks):
    sys.path.insert(0, str(PROJECT))
    from app.settings.settings import settings
    selected = [c for c in chunks if c["platform"] == platform]
    # 内容变更建新集合，不覆盖旧集合，也不遗留已删除片段；保留旧版便于对照。
    digest = hashlib.sha256(json.dumps(selected, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]
    settings.chroma_preset_path = PROJECT / "app/rag/backends/chroma_db/official_ecom"
    settings.chroma_collection_name = f"official_{platform}_{digest}"
    from app.rag.chunk import Chunk
    from app.rag.embedding import embedding
    from app.rag.backends.chroma_db import ecom_chroma_backend as backend
    vectors = embedding.embed([Chunk(c["chunk_id"], c["title"], c["section"],
                                     c["title"] + "\n" + c["section"] + "\n" + c["content"])
                               for c in selected])
    backend.chroma.upsert(ids=[c["chunk_id"] for c in selected], embeddings=vectors,
                          documents=[c["content"] for c in selected],
                          metadatas=[{"docs": c["title"], "section": c["section"],
                                      "platform": platform, "source_url": c["source_url"]}
                                     for c in selected])
    backend._build_and_save_bm25_cache()
    config = {"CHROMA_PRESET_PATH": str(settings.chroma_preset_path),
              "CHROMA_COLLECTION_NAME": settings.chroma_collection_name}
    write_json(ROOT / "exports" / platform / "index_config.json", config)
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", choices=["pdd", "jd"], help="会调用 embedding API；不传则纯离线")
    args = parser.parse_args()
    chunks = prepare()
    if args.index:
        index(args.index, chunks)
