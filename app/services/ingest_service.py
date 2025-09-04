from datetime import datetime
from typing import Tuple, Dict, Optional, List
from pathlib import Path

from app.utils.ids import new_id
from app.domain.contracts import (
    IngestService, UserID, FileID, TaskID, TaskProgress, JobStage,
    FileMeta, FileType, Chunk, ChunkMeta, SegmentType, NeighborInfo)

from app.core.settings import get_settings
from app.core.limits import check_limits
from app.storage.files import compute_doc_id, guess_file_type, save_original
from app.storage.pdf_inspect import inspect_pdf
from app.storage.docx_pdf import docx_to_pdf
from app.storage.text_extractor import iter_pdf_page_texts, make_text_chunks_from_pages
from app.storage.ocr_extractor import pdf_ocr_to_pages
from app.storage.pdf_scan_detect import scan_pdf_tables_and_text
from app.storage.pdf_visual import harvest_visual_cands
from app.storage.vectorstore_chroma import upsert_chunks, upsert_title
from app.llm.embeddings import get_embedding_client
from app.llm.vision import analyze_image, table_json_to_text
from app.llm.embeddings import get_embedding_client
from app.utils.ids import new_id



# —— 内存态“数据库”（演示用） ——
_TASKS: Dict[TaskID, dict] = {}  # task_id -> {start, duration, doc_id, ...}
_FILES_RAW: Dict[FileID, dict] = {}  # file_id -> {doc_id, owner_id, filename, ...}
_FILE_META: Dict[FileID, FileMeta] = {}  # file_id -> FileMeta
_DOC_TO_FILE: Dict[str, FileID] = {}  # doc_id -> file_id
_DOC_READY: Dict[str, bool] = {}  # doc_id -> 是否“索引完成”（此阶段模拟）
_DOC_CHUNKS: Dict[str, List[Chunk]] = {}  # NEW: doc_id -> 切片列表（供问答检索）


class SimpleIngestService(IngestService):
    """上传编排服务（演示阶段）。

    负责：
    - 计算 doc_id 去重；
    - 限额校验；
    - 落盘原始文件；
    - 若为 PDF，读取页数并粗分 文本/扫描；
    - 产出 file_id / task_id，并在内存里记录 FileMeta 与任务。
    - 若为文本型 PDF（或无法判定但先按文本处理），在上传时同步做“文本抽取 + 切片”，
      结果存进 _DOC_CHUNKS，供问答使用。
    - 注意：这里仍然用“8 秒计时器”来显示进度；真实环境将改为异步队列逐阶段推进。
    """

    def schedule_ingest(self, owner_id: UserID, filename: str, mime: str, file_bytes: bytes) -> Tuple[FileID, TaskID]:

        settings = get_settings()
        now = datetime.utcnow()

        # 1) 计算 doc_id（sha256），用于去重
        doc_id = compute_doc_id(file_bytes)
        if doc_id in _DOC_TO_FILE:
            # 命中去重：复用旧的 file_id，并返回一个“秒完成”的任务
            file_id = _DOC_TO_FILE[doc_id]
            task_id: TaskID = new_id("task_")
            # 命中去重：返回一个“秒完成”的任务
            _TASKS[task_id] = {"start": now, "duration": 0.0, "ready": True, "doc_id": doc_id}
            return file_id, task_id

        # 2) 生成 file_id，构建初始 FileMeta（先不填页数），并做限额校验
        file_id: FileID = new_id("file_")
        ftype: FileType = guess_file_type(filename, mime)
        meta = FileMeta(
            file_id=file_id,
            doc_id=doc_id,
            owner_id=owner_id,
            filename=filename,
            mime=mime or "",
            size_bytes=len(file_bytes),
            file_type=ftype,
            created_at=now,
            page_count=None,
            image_count=None,
        )
        lim = check_limits(meta, settings)
        if not lim.ok:
            raise ValueError(f"limit_violation:{lim.reason}")

        # 3) 落盘原始文件（data/original/{owner}/{doc}/original.ext）
        saved_path: Path = save_original(owner_id, doc_id, filename, file_bytes)


        # 4) 若为 PDF：用 PyMuPDF 读取页数，并粗略判断是否扫描件
        # 注意：guess_file_type() 对 ".pdf" 先归类为 pdf_text，若判定扫描则改成 pdf_scan
        if saved_path.suffix.lower() == ".pdf":
            total, is_scanned, stats = inspect_pdf(saved_path)
        # 这里 total=0 可能是 pymupdf 不可用或文件异常；不抛错，仅记录
        if total and total > 0:
            meta.page_count = total
        if is_scanned:
            meta.file_type = FileType.pdf_scan
        # 也可以把 stats 打日志，便于调试（此处略）

        # 5) 记录映射 & 元信息
        _FILES_RAW[file_id] = {
            "doc_id": doc_id,
            "owner_id": owner_id,
            "filename": filename,
            "mime": mime,
            "size": len(file_bytes),
            "path": str(saved_path),
        }
        _DOC_TO_FILE[doc_id] = file_id
        _FILE_META[file_id] = meta
        _DOC_READY[doc_id] = False

        # 6) 文本型 PDF：同步文本抽取与切片
        if meta.file_type in (FileType.pdf_text,):
            pages = list(iter_pdf_page_texts(saved_path))
            # 生成“窗口化切片”（返回字典列表）
            raw_chunks = make_text_chunks_from_pages(doc_id, pages, window_min=300, window_max=500, overlap=100)
            chunks: List[Chunk] = []
            for rc in raw_chunks:
                chunk_id = new_id("chk_")
                meta_obj = ChunkMeta(
                    chunk_id=chunk_id,  # 如果你这里写过，记得：chunk_id 其实在 Chunk() 里，不在 ChunkMeta 里
                    doc_id=doc_id,
                    segment_type=SegmentType.text,
                    page_from=rc["page_from"],
                    page_to=rc["page_to"],
                    title_guess="",
                    page_estimated=False,
                    appendix_flag=False,
                    # ↓↓↓ 这四个都可以不写，默认 None。先保留做占位，后面接邻近召回时再填。
                    neighbors_prev_page=None,
                    neighbors_next_page=None,
                    neighbors_prev_chunk_id=None,
                    neighbors_next_chunk_id=None,
                )
                chunks.append(Chunk(chunk_id=chunk_id, text=rc["text"], metadata=meta_obj))
            _DOC_CHUNKS[doc_id] = chunks
            # try:
            embedder = get_embedding_client()  # 若未配置会抛异常，我们捕获后允许回退
            texts = [c.text for c in chunks]
            vecs = embedder.embed_texts(texts)
            upsert_chunks(owner_id, chunks, vecs)
            upsert_title(owner_id, doc_id, title_text=filename, filename=filename)
            # except Exception as e:
            #     # 记录一下即可（此阶段允许无向量索引，问答会回退到关键词）
            #     print("向量写入失败")
            #     pass

        # 6.1) 扫描型 PDF：逐页 OCR → 切片（与文本型相同窗口策略）
        elif meta.file_type in (FileType.pdf_scan,):
            page_texts, table_crops = scan_pdf_tables_and_text(str(saved_path), dpi=300)

            chunks: list[Chunk] = []

            # 2) 文本页 → 切片（简单：一页一片，或拼成长段再按窗口切）
            for pt in page_texts:
                text = pt.text.strip()
                if not text:
                    continue
                # 可按你现有的窗口/重叠策略，这里先一页一片
                chunk_id = new_id("seg_")
                meta = ChunkMeta(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    segment_type=SegmentType.text,
                    page_from=pt.page,
                    page_to=pt.page,
                    title_guess="",
                    page_estimated=False,
                    appendix_flag=False,
                )
                chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=meta))

            # 3) 表格裁片 → 视觉 LLM 判类 + 文本化
            for tb in table_crops:
                obj = analyze_image(tb.png_bytes)  # {kind,title,text,table:{...}}
                kind = str(obj.get("kind", "table")).lower()
                title = obj.get("title", "") or ""
                table = obj.get("text")
                if kind == "table":
                    payload_text = table_json_to_text(str(table))
                    seg_type = SegmentType.table
                    class_bonus_hint = "table"
                else:
                    # 扫描件里通常不会识别成 chart/figure，但保留兼容
                    payload_text = obj.get("text", "") or title
                    seg_type = SegmentType.figure if kind not in ("chart", "photo") else SegmentType.chart
                    class_bonus_hint = "chart" if kind == "chart" else "figure"

                meta = ChunkMeta(
                    chunk_id=chunk_id,
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
            DOC_CHUNKS[doc_id] = chunks

        # 7) 新建任务（计时器模拟阶段推进）
        task_id: TaskID = new_id("task_")
        _TASKS[task_id] = {"start": now, "duration": 8.0, "doc_id": doc_id}
        return file_id, task_id

    def get_task_progress(self, task_id: TaskID) -> TaskProgress:
        rec = _TASKS.get(task_id)
        if not rec:
            return TaskProgress(task_id=task_id, stage=JobStage.failed, phase_percent=0, error="task_not_found")

        if rec.get("ready"):
            return TaskProgress(task_id=task_id, stage=JobStage.ready, phase_percent=100, eta_seconds=0)

        start: datetime = rec["start"]
        dur: float = rec["duration"]
        elapsed = (datetime.utcnow() - start).total_seconds()
        ratio = max(0.0, min(1.0, elapsed / dur))
        percent = int(ratio * 100)

        # 阶段划分（演示）
        if ratio >= 1.0:
            doc_id: Optional[str] = rec.get("doc_id")
            if doc_id:
                _DOC_READY[doc_id] = True
            return TaskProgress(task_id=task_id, stage=JobStage.ready, phase_percent=100, eta_seconds=0)

        if percent < 20:
            stage = JobStage.converting
        elif percent < 50:
            stage = JobStage.chunking
        elif percent < 80:
            stage = JobStage.embedding
        else:
            stage = JobStage.indexing

        eta_left = int((1.0 - ratio) * dur)
        return TaskProgress(task_id=task_id, stage=stage, phase_percent=percent, eta_seconds=eta_left, queue_len=0)

    # 新增：查询文件元信息
    def get_file_meta(self, file_id: FileID) -> Optional[FileMeta]:
        return _FILE_META.get(file_id)


# 暴露给路由层使用的单例和“切片仓库”访问（给 QA 用）
ingest_service = SimpleIngestService()
# 供其他模块（qa_service）读取解析结果（演示期跨模块引用，之后会抽到存储层）
DOC_CHUNKS = _DOC_CHUNKS
FILES_RAW = _FILES_RAW
DOC_TO_FILE = _DOC_TO_FILE
