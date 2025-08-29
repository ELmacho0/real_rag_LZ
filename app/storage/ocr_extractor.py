"""
app/storage/ocr_extractor.py

职责：
- 将 PDF 每页转为图像（pdf2image），
- 调用 RapidOCR 做 OCR，
- 返回 [(page_no, text), ...] 供后续切片。

说明：
- 这是一版“可用优先”的实现；后续可以：
  * 提升 DPI 或自适应（清晰度 vs 速度权衡）
  * 按页并行
  * 复用 OCR 实例
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import os

# pdf → images
from pdf2image import convert_from_path

# RapidOCR（onnxruntime 后端）
try:
    from rapidocr_onnxruntime import RapidOCR

    print("rapidocr_onnxruntime 按上了")
except Exception as e:  # pragma: no cover
    RapidOCR = None  # 允许上层优雅降级
    print("rapidocr_onnxruntime 没按上")


def pdf_ocr_to_pages(
    pdf_path: str | Path,
    dpi: int = 200,
    poppler_path: Optional[str] = None,
) -> List[Tuple[int, str]]:
    """
    对 PDF 逐页 OCR，返回 [(page_no, text), ...]。

    参数：
    - dpi: 图像清晰度；200 在速度/准确度之间较平衡。
    - poppler_path: Windows 下需要；若为空会尝试读取环境变量 POPPLER_PATH。
    """
    if RapidOCR is None:
        # 未安装 RapidOCR，返回空；上层据此跳过 OCR 流程
        return []

    p = Path(pdf_path)
    if not p.exists():
        return []

    # 解析 Poppler 路径（Windows）
    if poppler_path is None:
        poppler_path = os.environ.get("POPPLER_PATH")

    # 1) PDF 转图片（PIL.Image 列表）
    try:
        images = convert_from_path(str(p), dpi=dpi, poppler_path=poppler_path)
    except Exception as e:
        # 无 Poppler 或 PDF 异常等；返回空让上层优雅退避
        return []

    # 2) 初始化 OCR（模型会按需下载缓存到用户目录）
    ocr = RapidOCR()

    results: List[Tuple[int, str]] = []
    for idx, img in enumerate(images, start=1):
        # RapidOCR 支持直接喂 numpy/PIL；这里直接传 PIL.Image
        try:
            res, elapse = ocr(img)
        except Exception:
            res = None
        # res 形如 [[box, text, score], ...]
        if not res:
            results.append((idx, ""))
            continue
        page_lines = []
        for item in res:
            # 防御式解析
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            line_text = item[1] or ""
            page_lines.append(str(line_text))
        # 合并为一页文本（简单按换行拼接）
        page_text = "\n".join(page_lines).strip()
        results.append((idx, page_text))

    return results