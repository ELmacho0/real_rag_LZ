# app/storage/excel_extractor.py
from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import json
from openpyxl import load_workbook

from app.domain.contracts import Chunk, ChunkMeta, SegmentType
from app.utils.ids import new_id


def _sheet_preview(ws, max_rows: int = 10) -> Tuple[list[str], list[list[str]]]:
    rows = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i >= max_rows:
            break
        rows.append(["" if v is None else str(v) for v in row])
    headers = rows[0] if rows else []
    data = rows[1:] if len(rows) > 1 else []
    return headers, data


def parse_excel_to_chunks(file_path: str, doc_id: str, filename: str) -> List[Chunk]:
    """
    每个 Sheet 生成 1 个切片：
    - 文本内容 = 简述（字段/行数/可能的时间列）+ 少量预览（序列化为 JSON 保存在 metadata）
    - metadata 中附带 sheet_name、row_count、sheet_preview_json（字符串）
    """
    p = Path(file_path)
    wb = load_workbook(p, data_only=True, read_only=True)
    chunks: List[Chunk] = []
    for ws in wb.worksheets:
        sheet_name = ws.title or "Sheet"
        headers, data = _sheet_preview(ws, max_rows=10)
        row_count = ws.max_row or 0

        # 粗略猜测“可能的时间列”（包含年-月/年/日期样式的表头）
        guess_time_cols = [h for h in headers if any(t in (h or "") for t in ["年", "月", "日期", "time", "date"])]

        # 生成简述文本（用于向量/检索）
        summary_lines = [
            f"Excel Sheet: {sheet_name}",
            f"字段: {', '.join(h for h in headers if h)[:300]}",
            f"行数: {row_count}",
        ]
        if guess_time_cols:
            summary_lines.append(f"可能的时间字段: {', '.join(guess_time_cols)}")
        text = "\n".join(summary_lines)

        # 预览 JSON（作为 metadata 的字符串存储，便于 UI 展示或调试）
        preview_obj = {"headers": headers, "rows": data}
        preview_json = json.dumps(preview_obj, ensure_ascii=False)

        chunk_id = new_id("chk_")
        meta = ChunkMeta(
            doc_id=doc_id,
            segment_type=SegmentType.excel_sheet,
            page_from=None,
            page_to=None,
            title_guess=sheet_name,
            page_estimated=False,
            appendix_flag=False,
            neighbors_prev_page=None,
            neighbors_next_page=None,
            neighbors_prev_chunk_id=None,
            neighbors_next_chunk_id=None,
        )
        # 将 Sheet 相关的原子字段直接塞进 metadata（Chroma 仅支持原子类型）
        md = meta.model_dump(exclude_none=True)
        md.update({
            "sheet_name": sheet_name,
            "sheet_rows": int(row_count),
            "sheet_preview_json": preview_json,
            "filename": filename,
        })
        chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=ChunkMeta(**md)))
    return chunks
