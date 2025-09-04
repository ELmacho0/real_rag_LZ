# app/services/ingest_service.py 片段（示意）
from app.storage.pdf_scan_detect import scan_pdf_tables_and_text
from app.llm.vision import analyze_image, table_json_to_text
from app.llm.embeddings import get_embedding_client
from app.storage.vectorstore_chroma import upsert_chunks, upsert_title
from app.domain.contracts import Chunk, ChunkMeta, SegmentType
from app.utils.ids import new_id

def _ingest_pdf_scanned(owner_id: str, doc_id: str, filename: str, pdf_path: str):
    # 1) 扫描件专用：拿“表格裁片 + 遮蔽表格后的页面文本”
    page_texts, table_crops = scan_pdf_tables_and_text(pdf_path, dpi=300)

    chunks: list[Chunk] = []

    # 2) 文本页 → 切片（简单：一页一片，或拼成长段再按窗口切）
    for pt in page_texts:
        text = pt.text.strip()
        if not text:
            continue
        # 可按你现有的窗口/重叠策略，这里先一页一片
        meta = ChunkMeta(
            doc_id=doc_id,
            segment_type=SegmentType.text,
            page_from=pt.page,
            page_to=pt.page,
            title_guess="",
            page_estimated=False,
            appendix_flag=False,
        )
        chunks.append(Chunk(chunk_id=new_id("seg_"), text=text, metadata=meta))

    # 3) 表格裁片 → 视觉 LLM 判类 + 文本化
    for tb in table_crops:
        obj = analyze_image(tb.png_bytes)  # {kind,title,text,table:{...}}
        kind = str(obj.get("kind","table")).lower()
        title = obj.get("title","") or ""
        table = obj.get("table") or {"headers":[],"rows":[],"notes":[]}
        if kind == "table":
            payload_text = table_json_to_text(table)
            seg_type = SegmentType.table
            class_bonus_hint = "table"
        else:
            # 扫描件里通常不会识别成 chart/figure，但保留兼容
            payload_text = obj.get("text","") or title
            seg_type = SegmentType.figure if kind not in ("chart","photo") else SegmentType.chart
            class_bonus_hint = "chart" if kind=="chart" else "figure"

        meta = ChunkMeta(
            doc_id=doc_id,
            segment_type=seg_type,
            page_from=tb.page,
            page_to=tb.page,
            title_guess=title,
            page_estimated=False,
            appendix_flag=False,
        )
        chunks.append(Chunk(chunk_id=new_id("seg_"), text=payload_text, metadata=meta))

    # 4) 入向量库
    if chunks:
        embedder = get_embedding_client()
        vecs = embedder.embed_texts([c.text for c in chunks])
        upsert_chunks(owner_id, chunks, vecs)

        # 5) 将非文字切片写入标题索引（便于“标题命中 → 直达该切片”）
        for c in chunks:
            if c.metadata.segment_type in (SegmentType.table, SegmentType.chart) and c.metadata.title_guess:
                upsert_title(
                    owner_id=owner_id,
                    doc_id=doc_id,
                    title_text=c.metadata.title_guess,
                    filename=filename,
                    # 如果你已把 upsert_title 增强为可写元数据：
                    title_kind=c.metadata.segment_type.value,
                    target_id=c.chunk_id,
                )

    # 6) 写入内存缓存，方便调试预览
    DOC_CHUNKS[doc_id] = DOC_CHUNKS.get(doc_id, []) + chunks
