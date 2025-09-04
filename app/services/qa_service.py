from typing import Dict, Optional, List
from collections import defaultdict

from app.utils.ids import new_id
from app.domain.contracts import QAService, UserID, QARequest, AnswerID, QAAnswer, QAStage, Citation
from app.services.ingest_service import DOC_CHUNKS, FILES_RAW, DOC_TO_FILE
from app.llm.embeddings import get_embedding_client
from app.storage.vectorstore_chroma import query_topk, query_titles, get_chunks_by_doc, get_by_ids, get_doc_segments
from app.retrieval.rewrite import generate_rewrites
from app.retrieval.rerank import get_rerank_client
from app.retrieval.fusion import Cand, score, is_unreliable_top
from app.retrieval.keyword import tokenize, score_text  # 作为最终兜底
from app.core.settings import get_settings
_ANSWERS: Dict[AnswerID, QAAnswer] = {}


class SimpleQAService(QAService):
    def submit_query(self, owner_id: UserID, query: str, top_k: int = 8, session_id: Optional[str] = None, final_k: int | None = None) -> AnswerID:
        answer_id: AnswerID = new_id("ans_")

        # 0) 生成改写 Q*
        queries = generate_rewrites(query, k=4)  # [Q, R1..R4]（占位实现）
        if not queries:
            queries = [query]

        # 1) 多路召回（收集候选，不使用 emb 相似度打最终分）
        cand_map: Dict[str, Cand] = {}  # chunk_id -> Cand（合并多路命中）
        rewrite_weight_for = {}  # 文本 query -> 权重
        if queries:
            rewrite_weight_for[queries[0]] = 1.0
            for q in queries[1:]:
                rewrite_weight_for[q] = 0.8

        # 1.1 内容向量召回
        try:
            embedder = get_embedding_client()
            for q in queries:
                qvec = embedder.embed_texts([q])[0]
                hits = query_topk(owner_id, qvec, top_k=4) or []
                for h in hits:
                    md = h.get("metadata", {})
                    cid = h["id"]
                    c = cand_map.get(cid)
                    if not c:
                        c = Cand(
                            chunk_id=cid, doc_id=md.get("doc_id", ""), text=h.get("document", ""),
                            page_from=md.get("page_from"), emb_sim=1.0 - float(h.get("distance", 1.0)),
                            rewrite_weight=rewrite_weight_for.get(q, 0.8), class_bonus=0.0
                        )
                        cand_map[cid] = c
                    else:
                        c.multi_hit += 1
                        c.rewrite_weight = max(c.rewrite_weight, rewrite_weight_for.get(q, 0.8))
        except Exception:
            pass

        # 1.2 标题索引召回（仅接受带 target_id 的标题，如 excel_sheet/table/chart）。
        try:
            embedder = get_embedding_client()
            qvec_main = embedder.embed_texts([queries[0]])[0]
            title_hits = query_titles(owner_id, qvec_main, top_k=2) or []
            target_ids: List[str] = []
            for th in title_hits:
                md = th.get("metadata", {})
                target_id = md.get("target_id")
                title_kind = md.get("title_kind")
                # 仅当具备 target 切片时才纳入（避免 PDF 文件名噪声）
                if not target_id or not title_kind:
                    continue
                target_ids.append(target_id)
            if target_ids:
                chunks = get_by_ids(owner_id, target_ids)
                for s in chunks:
                    smd = s.get("metadata", {})
                    cid = s["id"]
                    c = cand_map.get(cid)
                    if not c:
                        c = Cand(
                            chunk_id=cid, doc_id=smd.get("doc_id", ""), text=s.get("document", ""),
                            page_from=smd.get("page_from"), emb_sim=None,
                            rewrite_weight=0.6, class_bonus=(0.06 if smd.get("segment_type") == "table" else (
                                0.03 if smd.get("segment_type") == "chart" else 0.0)),
                            title_hit=True
                        )
                        cand_map[cid] = c
                    else:
                        c.title_hit = True
                        c.rewrite_weight = max(c.rewrite_weight, 0.6)
                        c.multi_hit += 1
        except Exception:
            pass

        # 1.3 邻近强制召回：仅“之后最近的 table/chart”
        #    对前 Top-N=6 个已存在候选（按当前收集量排序即可）逐一处理
        base_cands = list(cand_map.values())[:6]
        for base in base_cands:
            if not base.doc_id or base.page_from is None:
                continue
            segs = get_doc_segments(owner_id, base.doc_id)
            # 在同一文档中过滤 table/chart，且 page_from > base.page_from，取最小差值者
            # 待修改，貌似只会召回一张表，并且如果文档少于10页坑无法成功召回附表
            next_tc = None
            next_gap = 10 ** 9
            for s in segs:
                md = s.get("metadata", {})
                if md.get("segment_type") not in ("table", "chart"):
                    continue
                p = md.get("page_from")
                if p is None or p <= base.page_from:
                    continue
                gap = p - base.page_from
                if gap < next_gap:
                    next_gap = gap
                    next_tc = s
            if next_tc:
                smd = next_tc.get("metadata", {})
                cid = next_tc["id"]
                c = cand_map.get(cid)
                if not c:
                    c = Cand(
                        chunk_id=cid, doc_id=smd.get("doc_id", ""), text=next_tc.get("document", ""),
                        page_from=smd.get("page_from"), emb_sim=None,
                        rewrite_weight=0.6, class_bonus=(0.06 if smd.get("segment_type") == "table" else 0.03),
                        neighbor_boost=True
                    )
                    cand_map[cid] = c
                else:
                    c.neighbor_boost = True
                    c.rewrite_weight = max(c.rewrite_weight, 0.6)
                    c.multi_hit += 1

        # 2) 聚合候选，送入 reranker
        candidates = list(cand_map.values())
        if not candidates:
            _ANSWERS[answer_id] = QAAnswer(answer_id=answer_id, stage=QAStage.no_answer, text=None, citations=[],
                                           no_answer_reason="no_candidates")
            return answer_id

        rerank = get_rerank_client()
        docs = [c.text for c in candidates]
        topn = min(len(docs), get_settings().RERANK_TOPN if hasattr(get_settings(), "RERANK_TOPN") else 50)
        try:
            rr = rerank.rerank(query=queries[0], documents=docs, top_n=topn, return_documents=False)
            # 将 rerank 分数写回到候选（按 index 对齐）
            for item in rr:
                idx = item["index"]
                if 0 <= idx < len(candidates):
                    candidates[idx].rel = float(item.get("score", 0.0))
        except Exception:
            # rerank 失败：rel 全 0，按其他 boost 兜底
            pass

        # 3) 融合打分 & 无答案规则
        ranked = sorted(candidates, key=score, reverse=True)

        s = get_settings()
        final_k = final_k or getattr(s, "ANSWER_FINAL_K", 4)  # 如果你把 final_k 参数传进来
        final_k = max(1, min(final_k, len(ranked)))

        # 选前 K 个做引用；（也可以拼接多段，这里先保持简洁）
        top_main = ranked[0]
        top_k_items = ranked[:final_k]

        citations = []
        for item in top_k_items:
            file_id = DOC_TO_FILE.get(item.doc_id, "")
            filename = FILES_RAW.get(file_id, {}).get("filename", "unknown")
            citations.append(Citation(
                doc_id=item.doc_id,
                chunk_id=item.chunk_id,
                filename=filename,
                page_hint=(f"p.{item.page_from}" if item.page_from else None),
            ))

        preview = (top_main.text or "").strip().replace("\n", " ")
        if len(preview) > 1000: preview = preview[:1000] + "…"

        _ANSWERS[answer_id] = QAAnswer(
            answer_id=answer_id,
            stage=QAStage.done,
            text=f"【融合+Rerank】{preview}",
            citations=citations,
            no_answer_reason=None
        )
        return answer_id

    def get_answer(self, answer_id: AnswerID) -> QAAnswer:
        ans = _ANSWERS.get(answer_id)
        if not ans:
            return QAAnswer(answer_id=answer_id, stage=QAStage.failed, text=None, citations=[],
                            no_answer_reason="answer_not_found")
        return ans


qa_service = SimpleQAService()
