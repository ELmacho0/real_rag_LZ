# app/storage/vectorstore_chroma.py
from __future__ import annotations
from typing import List, Dict, Any
import chromadb
from app.llm.embeddings import get_embedding_client
from app.core.settings import get_settings
from app.domain.contracts import Chunk

_client = None


def get_client():
    global _client
    if _client is None:
        s = get_settings()
        # 持久化到本地（DuckDB+Parquet），路径见 s.CHROMA_PATH
        _client = chromadb.PersistentClient(path=s.CHROMA_PATH)
    return _client


def get_or_create_collection(name: str):
    client = get_client()
    print("尝试创建集合")
    try:
        col = client.get_collection(name)
    except Exception:
        # 余弦相似度（默认即 cosine）；不同版本写法会略有不同
        col = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    return col


def _drop_none(md: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in (md or {}).items() if v is not None}


def upsert_chunks(owner_id: str, chunks: List[Chunk], embeddings: List[List[float]]):
    if not chunks:
        return
    if len(chunks) != len(embeddings):
        raise ValueError("embeddings size mismatch")

    col = get_or_create_collection(f"kb_{owner_id}")

    ids = [c.chunk_id for c in chunks]
    docs = [c.text for c in chunks]
    metas: List[Dict[str, Any]] = []
    for c in chunks:
        raw = c.metadata.model_dump(exclude_none=True) if c.metadata else {}
        raw["doc_id"] = c.metadata.doc_id if c.metadata else None
        metas.append(_drop_none(raw))  # 只去 None，剩下都是原子类型，Chroma 可接受

    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)


def query_topk(owner_id: str, query_vec: List[float], top_k: int = 8):
    col = get_or_create_collection(f"kb_{owner_id}")
    print("准备计算问题相似度")
    res = col.query(query_embeddings=[query_vec], n_results=top_k, include=["documents", "metadatas", "distances"])
    # 统一提取第一条查询的返回
    out = []
    if res and res.get("ids") and res["ids"][0]:
        print("開始問了")
        for idx, _id in enumerate(res["ids"][0]):
            out.append({
                "id": _id,  # 直接从 res["ids"] 取，不需要 include
                "document": res["documents"][0][idx],
                "metadata": res["metadatas"][0][idx],
                "distance": res["distances"][0][idx],
            })
    return out


def get_or_create_title_collection(owner_id: str):
    name = f"kb_title_{owner_id}"
    client = get_client()
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert_title(owner_id: str, doc_id: str, title_text: str, filename: str, *, title_kind: str | None = None, target_id: str | None = None):
    if not title_text:
        return
    col = get_or_create_title_collection(owner_id)
    embedder = get_embedding_client()
    vec = embedder.embed_texts([title_text])[0]
    meta = {"doc_id": doc_id, "filename": filename}
    if title_kind: meta["title_kind"] = title_kind
    if target_id: meta["target_id"] = target_id
    col.upsert(
        ids=[f"title_{doc_id}_{(target_id or 'doc').replace(' ','_')}"],
        documents=[title_text],
        metadatas=[meta],
        embeddings=[vec],
    )


def query_titles(owner_id: str, query_vec: list[float], top_k: int = 3):
    col = get_or_create_title_collection(owner_id)
    res = col.query(query_embeddings=[query_vec], n_results=top_k, include=["documents", "metadatas", "distances"])
    out = []
    if res and res.get("ids") and res["ids"][0]:
        for i, _id in enumerate(res["ids"][0]):
            out.append({
                "id": _id,
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            })
    return out


def get_chunks_by_doc(owner_id: str, doc_id: str, limit: int = 4):
    """按 doc_id 取若干切片；用于标题命中后的“正文补充证据”。"""
    col = get_or_create_collection(f"kb_{owner_id}")
    res = col.get(where={"doc_id": doc_id}, limit=limit, include=["documents", "metadatas"])
    out = []
    if res and res.get("ids"):
        for i, _id in enumerate(res["ids"]):
            out.append({
                "id": _id,
                "document": res["documents"][i],
                "metadata": res["metadatas"][i],
            })
    return out


def get_by_ids(owner_id: str, ids: list[str]):
    col = get_or_create_collection(f"kb_{owner_id}")
    if not ids:
        return []
    res = col.get(ids=ids, include=["documents", "metadatas"])
    out = []
    if res and res.get("ids"):
        for i, _id in enumerate(res["ids"]):
            out.append({
                "id": _id,
                "document": res["documents"][i],
                "metadata": res["metadatas"][i],
            })
    return out


def get_doc_segments(owner_id: str, doc_id: str):
    """取某文档的全部切片（用于邻近检索时本地筛选）。"""
    col = get_or_create_collection(f"kb_{owner_id}")
    res = col.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
    out = []
    if res and res.get("ids"):
        for i, _id in enumerate(res["ids"]):
            out.append({
                "id": _id,
                "document": res["documents"][i],
                "metadata": res["metadatas"][i],
            })
    return out
