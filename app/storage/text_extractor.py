"""
app/storage/text_extractor.py

职责：
1) 从 PDF 提取“每页文本”（基于 PyMuPDF）。
2) 将多页文本合并后，按窗口大小做“滑动切片”（保留页码范围等元数据）。

说明：
- 这是“文本型 PDF”的流程；扫描 PDF 以后走 OCR（到时另写一个 ocr_extractor.py）。
- 这里的“窗口切片”是最简单可靠的方式，后续可以换成基于段落/标题的更智能切分。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Optional

try:
    import fitz as _fitz
    if not hasattr(_fitz, "open"):
        raise ImportError("wrong_fitz_on_path")
    fitz = _fitz
except Exception:
    fitz = None  # 上层应优雅降级/跳过


# -------- 文本抽取 --------

def iter_pdf_page_texts(path: str | Path) -> Iterable[Tuple[int, str]]:
    """
    逐页返回 (page_number, text)。
    - page_number 从 1 开始，便于和用户看到的页码一致。
    - 若未安装 PyMuPDF 或文件异常，则返回空迭代器。
    """
    if fitz is None:
        return
    p = Path(path)
    if not p.exists():
        return
    doc = fitz.open(p)
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            yield (i + 1, text)
    finally:
        doc.close()


# -------- 切片（窗口化） --------

def make_text_chunks_from_pages(
        doc_id: str,
        pages: List[Tuple[int, str]],
        window_min: int = 300,
        window_max: int = 500,
        overlap: int = 100,
):
    """
    将多页文本合并后做“滑动窗口切片”，并保留该切片覆盖的页码范围。
    返回值：List[dict]，每个 dict 包含：
        {
          "text": "切片文本",
          "page_from": 12,
          "page_to": 13
        }
    说明：
    - 为简化，这里只是一个简单的“尽量接近 window_max”的贪心策略：
      * 累加文本直到接近 window_max；若不足 window_min 且还有下一页，就继续累加；
      * 形成一个切片后，保留末尾 overlap 个字符作为下一片的开头（制造重叠）。
    - 新增：在冲洗 chunk 之前，去掉 chunk 中所有换行标记（同时处理实际换行\n/\r\n/\r 与字面“/n”）。
    - 后续可以用自然段分界、中文标点来“更优地断句”，但先以稳定为主。
    """
    chunks = []
    buf = ""
    cur_from: Optional[int] = None
    cur_to: Optional[int] = None

    def _remove_newlines(s: str) -> str:
        """移除所有换行相关字符与字面“/n”。"""
        # 先统一删除 Windows/Mac/Unix 换行，再删除字面“/n”
        return s.replace("\r\n", "").replace("\r", "").replace("\n", "").replace("/n", "")

    def flush_chunk():
        nonlocal buf, cur_from, cur_to
        # 新增：冲洗前移除换行
        cleaned = _remove_newlines(buf)
        text = cleaned.strip()
        if not text:
            return
        chunks.append({
            "text": text,
            "page_from": cur_from,
            "page_to": cur_to
        })
        # 生成下一窗口的“重叠前缀”
        if 0 < overlap < len(text):
            buf = text[-overlap:]
        else:
            buf = ""
        cur_from = None
        cur_to = None

    for page_no, page_text in pages:
        t = (page_text or "").strip()
        if not t:
            # 空页也要更新页码范围（以便页面跨度准确）
            if cur_from is None:
                cur_from = page_no
            cur_to = page_no
            continue

        # 若这是新窗口的起点，记录 page_from
        if cur_from is None:
            cur_from = page_no

        # 尝试加入本页文本，必要时分片
        idx = 0
        while idx < len(t):
            # 还差多少到上限
            room = max(0, window_max - len(buf))
            if room == 0:
                # 已经到上限，先冲洗为一个切片
                cur_to = page_no
                flush_chunk()
                continue
            # 取本页的一段，加入 buf
            take = t[idx: idx + room]
            buf += take
            idx += len(take)
            cur_to = page_no

            # 达到最小窗口要求，且再继续可能太长 → 冲洗
            if len(buf) >= window_min and (len(buf) >= window_max or idx >= len(t)):
                flush_chunk()

    # 最后一段若有剩余也要形成切片
    if buf.strip():
        if cur_from is None:
            cur_from = pages[-1][0] if pages else None
        if cur_to is None:
            cur_to = pages[-1][0] if pages else None
        flush_chunk()

    return chunks
