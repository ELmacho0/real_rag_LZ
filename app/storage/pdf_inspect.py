"""
app/storage/pdf_inspect.py


职责：基于 PyMuPDF 读取 PDF 的页数，并用非常轻量的启发式判断
“更像文本 PDF 还是扫描 PDF”。


注意：这不是严格算法，只是为了阶段性功能点亮；
后续可以加入更精细的判定逻辑（图像覆盖率、文字密度等）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

try:
    import fitz  # PyMuPDF 的包名是 pymupdf，但导入名叫 fitz
except Exception as e:  # pragma: no cover - 若环境缺少 pymupdf，不在此处抛出
    fitz = None  # 允许上层优雅降级


def inspect_pdf(path: str | Path, sample_pages: int = 5) -> Tuple[int, bool, Dict]:
    """
    读取 PDF 基本信息：
    - page_count: 总页数
    - is_scanned: 是否更像扫描件（粗略判断）
    - stats: 一些中间统计，便于调试


    启发规则（非常粗略，但足够在这个阶段用）：
    - 取前 N 页（默认 5 页，若页数不足则取全部）。
    - 如果一页 "可提取文本长度很短" 且 "存在图片对象"，把这页记为 scanned-like。
    - 若 sampled_pages 中 scanned-like 的比例 ≥ 0.6，则整体视为扫描 PDF。
    """
    p = Path(path)
    if fitz is None:
        # 没装 pymupdf：返回“未知页数、非扫描”（上层可以选择忽略）
        return 0, False, {"error": "pymupdf_not_available"}

    if not p.exists():
        return 0, False, {"error": "file_not_found"}

    doc = fitz.open(p)
    try:
        total = doc.page_count
        sample_n = min(sample_pages, total) if total > 0 else 0
        scanned_like = 0
        page_text_lens = []
        page_image_counts = []

        for i in range(sample_n):
            page = doc.load_page(i)
            # 1) 抽取纯文本（PyMuPDF 的快速方式）
            text = page.get_text("text") or ""
            text_len = len(text.strip())
            page_text_lens.append(text_len)

            # 2) 统计该页里是否有图片对象
            images = page.get_images(full=True) or []
            img_cnt = len(images)
            page_image_counts.append(img_cnt)

            # 3) 启发判定：几乎没有文本 + 至少有 1 张图 → 更像扫描页
            if text_len < 20 and img_cnt >= 1:
                scanned_like += 1

        is_scanned = False
        if sample_n > 0:
            is_scanned = (scanned_like / sample_n) >= 0.6

        stats = {
            "sampled": sample_n,
            "page_text_lens": page_text_lens,
            "page_image_counts": page_image_counts,
            "scanned_like_pages": scanned_like,
        }
        return total, is_scanned, stats
    finally:
        doc.close()
