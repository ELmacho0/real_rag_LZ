# app/storage/pdf_visual.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import io, math
import fitz  # PyMuPDF
import cv2, numpy as np
from pdf2image import convert_from_path
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
import re
from app.core.settings import get_settings

_ocr = RapidOCR()


@dataclass
class VisualCand:
    page: int
    kind_hint: str  # "table" | "image"（image 后续由 LLM 决定 chart/figure/photo）
    bbox_pdf: Tuple[float, float, float, float]  # (x1,y1,x2,y2) PDF 坐标
    crop_png: bytes


# —— 表格检测（简化自你的实现） ——
def _detect_table_boxes(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35 | 1, 15)
    vk = max(1, int(w * 0.018));
    hk = max(1, int(h * 0.015))
    ver_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    hor_k = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    v = cv2.dilate(cv2.erode(th, ver_k, iterations=2), ver_k, iterations=2)
    hline = cv2.dilate(cv2.erode(th, hor_k, iterations=2), hor_k, iterations=2)
    tbl = cv2.dilate(cv2.bitwise_or(v, hline), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
    cnts, _ = cv2.findContours(tbl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    min_area = int(0.01 * w * h)
    for c in cnts:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww * hh < min_area: continue
        if ww < int(0.12 * w) or hh < int(0.06 * h): continue
        boxes.append((x, y, x + ww, y + hh))
    # 合并近邻
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def _imencode_png(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.png', bgr)
    return buf.tobytes() if ok else b''


def _merge_boxes(boxes, iou_th=0.2, gap=8):
    # 简易近邻/IoU 合并，避免碎片过多
    boxes = boxes[:]
    changed = True
    while changed:
        changed = False
        new = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]: continue
            ax1, ay1, ax2, ay2 = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]: continue
                bx1, by1, bx2, by2 = boxes[j]
                # 邻近或交并比高就合并
                near = (abs(ax1 - bx1) <= gap or abs(ay1 - by1) <= gap or abs(ax2 - bx2) <= gap or abs(
                    ay2 - by2) <= gap)
                # IoU
                inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
                inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
                inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
                inter = inter_w * inter_h
                if inter > 0:
                    area_a = (ax2 - ax1) * (ay2 - ay1);
                    area_b = (bx2 - bx1) * (by2 - by1)
                    iou = inter / max(1, (area_a + area_b - inter))
                else:
                    iou = 0.0
                if near or iou >= iou_th:
                    ax1, ay1 = min(ax1, bx1), min(ay1, by1)
                    ax2, ay2 = max(ax2, bx2), max(ay2, by2)
                    used[j] = True
                    changed = True
            used[i] = True
            new.append((ax1, ay1, ax2, ay2))
        boxes = new
    return boxes


def _is_scanned_page(page: fitz.Page) -> bool:
    # 可提取文本很少，且向量文本块数量很少 → 扫描页
    txt = page.get_text("text") or ""
    if len(txt.strip()) >= get_settings().VISION_SCAN_TEXT_CHAR_TH:
        return False
    blocks = page.get_text("blocks") or []
    text_block_cnt = sum(1 for b in blocks if len(b) >= 5 and isinstance(b[4], str) and b[4].strip())
    return text_block_cnt == 0


def _ocr_text_boxes(pil_img: Image.Image) -> list[tuple[int, int, int, int]]:
    # RapidOCR 返回 [ [pts, text, score], ... ]，取外接矩形并做膨胀
    res, _ = _ocr(pil_img)
    boxes = []
    if not res:
        return boxes
    dil = max(1, get_settings().VISION_TEXT_DILATE_PX)
    for item in res:
        pts = item[0]  # 4 点
        xs = [int(p[0]) for p in pts];
        ys = [int(p[1]) for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        boxes.append((x1 - dil, y1 - dil, x2 + dil, y2 + dil))
    return _merge_boxes(boxes, iou_th=0.3, gap=6)


def _non_text_regions_from_ocr(bgr: np.ndarray, text_boxes: list[tuple[int, int, int, int]]) -> list[
    tuple[int, int, int, int]]:
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in text_boxes:
        x1 = max(0, x1);
        y1 = max(0, y1);
        x2 = min(w, x2);
        y2 = min(h, y2)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    # 非文字 = 背景 - 文本；用开闭操作连接大块
    non_text = cv2.bitwise_not(mask)
    non_text = cv2.morphologyEx(non_text, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    non_text = cv2.morphologyEx(non_text, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(non_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    min_area = int(get_settings().VISION_NON_TEXT_MIN_AREA * w * h)
    for c in cnts:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww * hh >= min_area:
            boxes.append((x, y, x + ww, y + hh))
    return _merge_boxes(boxes, iou_th=get_settings().VISION_NON_TEXT_MERGE_IOU, gap=12)


def harvest_visual_cands(pdf_path: str) -> List[VisualCand]:
    s = get_settings()
    doc = fitz.open(pdf_path)
    out: List[VisualCand] = []
    try:
        # 为了避免重复渲染，每页渲染一次，复用给两种检测
        pil_pages = convert_from_path(pdf_path, dpi=s.VISION_PAGE_DPI)
        for pno, pil in enumerate(pil_pages, start=1):
            page = doc[pno - 1]
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            ph, pw = bgr.shape[:2]

            # 1) 表格候选（任何页面都启用）
            tboxes = _detect_table_boxes(bgr)
            for (x1, y1, x2, y2) in tboxes[: s.VISION_MAX_CANDS_PER_PAGE]:
                crop = bgr[y1:y2, x1:x2]
                rx = page.rect.width / pw;
                ry = page.rect.height / ph
                bx1, by1, bx2, by2 = x1 * rx, y1 * ry, x2 * rx, y2 * ry
                out.append(VisualCand(page=pno, kind_hint='table', bbox_pdf=(bx1, by1, bx2, by2),
                                      crop_png=_imencode_png(crop)))

            # 2) 图片/图表候选
            if _is_scanned_page(page):
                # —— 扫描页：用 OCR 文本掩膜反推“非文字区域” —— #
                text_boxes = _ocr_text_boxes(pil)
                img_boxes = _non_text_regions_from_ocr(bgr, text_boxes)
                # 如果出现“大块”非文字区域，一般就 1-2 个；避免碎片
                for (x1, y1, x2, y2) in img_boxes[: max(1, s.VISION_MAX_CANDS_PER_PAGE - len(tboxes))]:
                    crop = bgr[y1:y2, x1:x2]
                    rx = page.rect.width / pw;
                    ry = page.rect.height / ph
                    bx1, by1, bx2, by2 = x1 * rx, y1 * ry, x2 * rx, y2 * ry
                    out.append(VisualCand(page=pno, kind_hint='image', bbox_pdf=(bx1, by1, bx2, by2),
                                          crop_png=_imencode_png(crop)))
            else:
                # —— 文本页：保留 rawdict 内嵌图片 —— #
                rd = page.get_text('rawdict')
                if rd and 'blocks' in rd:
                    for b in rd['blocks']:
                        if b.get('type') == 1:
                            x1, y1, x2, y2 = map(int, b['bbox'])
                            sx = pw / page.rect.width;
                            sy = ph / page.rect.height
                            px1, py1, px2, py2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
                            if (px2 - px1) < 16 or (py2 - py1) < 16: continue
                            crop = bgr[py1:py2, px1:px2]
                            out.append(VisualCand(page=pno, kind_hint='image', bbox_pdf=tuple(b['bbox']),
                                                  crop_png=_imencode_png(crop)))

            # 3) 每页限流
            page_quota = s.VISION_MAX_CANDS_PER_PAGE
            # 过滤本页的候选并截断
            page_items = [c for c in out if c.page == pno]
            if len(page_items) > page_quota:
                # 优先保留表格，再保留面积大的图片
                def area(c: VisualCand):
                    # 以像素空间估算面积
                    bx1, by1, bx2, by2 = c.bbox_pdf
                    return (bx2 - bx1) * (by2 - by1)

                tables = [c for c in page_items if c.kind_hint == 'table']
                images = [c for c in page_items if c.kind_hint != 'table']
                images.sort(key=area, reverse=True)
                keep = (tables + images)[:page_quota]
                # 丢弃多余项
                out = [c for c in out if c.page != pno] + keep

        return out
    finally:
        doc.close()
