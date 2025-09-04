# app/storage/pdf_scan_detect.py
from __future__ import annotations
import os, re, io
from dataclasses import dataclass
from typing import List, Tuple
import cv2, numpy as np
from pdf2image import convert_from_path
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

# —— 参数（也可迁到 Settings，这里先内置默认） ——
DPI = 300
BIN_METHOD = "adaptive"     # "adaptive" | "otsu"
ADAPTIVE_BLOCK_SIZE = 35
ADAPTIVE_C = 15
GAUSS_BLUR = 3

VERT_KERNEL_RATIO = 0.018
HORZ_KERNEL_RATIO = 0.015
MORPH_ITER = 2

MIN_TABLE_AREA_RATIO = 0.01
MIN_W_RATIO = 0.12
MIN_H_RATIO = 0.06
MERGE_IOU_THRESHOLD = 0.15
MERGE_PIX_GAP = 12

_ocr = RapidOCR()

@dataclass
class PageText:
    page: int
    text: str   # 已经清洗合并的一段文本

@dataclass
class TableCrop:
    page: int
    png_bytes: bytes         # PNG 编码后的裁片

# ---------- 工具 ----------
def pil_to_bgr(img_pil: Image.Image):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def binarize(gray):
    if GAUSS_BLUR and GAUSS_BLUR > 1:
        gray = cv2.GaussianBlur(gray, (GAUSS_BLUR, GAUSS_BLUR), 0)
    if BIN_METHOD == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
            ADAPTIVE_BLOCK_SIZE | 1, ADAPTIVE_C
        )
    else:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return th

def extract_lines(bin_img):
    h, w = bin_img.shape[:2]
    vk = max(1, int(w * VERT_KERNEL_RATIO))
    hk = max(1, int(h * HORZ_KERNEL_RATIO))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    vertical = cv2.erode(bin_img, vertical_kernel, iterations=MORPH_ITER)
    vertical = cv2.dilate(vertical, vertical_kernel, iterations=MORPH_ITER)
    horizontal = cv2.erode(bin_img, horizontal_kernel, iterations=MORPH_ITER)
    horizontal = cv2.dilate(horizontal, horizontal_kernel, iterations=MORPH_ITER)
    table_lines = cv2.bitwise_or(vertical, horizontal)
    table_lines = cv2.dilate(table_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return table_lines

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0: return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / float(area_a + area_b - inter_area + 1e-6)

def merge_boxes(boxes, gap=MERGE_PIX_GAP, iou_th=MERGE_IOU_THRESHOLD):
    changed = True
    boxes = boxes[:]
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
                near = (abs(ax1 - bx1) <= gap or abs(ax2 - bx2) <= gap or
                        abs(ay1 - by1) <= gap or abs(ay2 - by2) <= gap)
                if near or iou((ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2)) >= iou_th:
                    ax1, ay1 = min(ax1, bx1), min(ay1, by1)
                    ax2, ay2 = max(ax2, bx2), max(ay2, by2)
                    used[j] = True
                    changed = True
            used[i] = True
            new.append((ax1, ay1, ax2, ay2))
        boxes = new
    return boxes

def detect_tables_on_bgr(bgr):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bin_img = binarize(gray)
    table_lines = extract_lines(bin_img)
    contours, _ = cv2.findContours(table_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    min_area = int(MIN_TABLE_AREA_RATIO * w * h)
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww * hh < min_area: continue
        if ww < int(MIN_W_RATIO * w) or hh < int(MIN_H_RATIO * h): continue
        boxes.append((x, y, x + ww, y + hh))
    boxes = merge_boxes(boxes)
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes

def mask_tables_on_bgr(bgr, boxes, margin_px=2):
    h, w = bgr.shape[:2]
    masked = bgr.copy()
    for (x1, y1, x2, y2) in boxes:
        x1 = max(0, x1 - margin_px)
        y1 = max(0, y1 - margin_px)
        x2 = min(w, x2 + margin_px)
        y2 = min(h, y2 + margin_px)
        cv2.rectangle(masked, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
    return masked

def clean_and_merge_lines(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    joined = " ".join(lines)
    parts = re.split(r"([。！？])", joined)
    rebuilt = []
    buf = ""
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        if not seg and not punct:
            continue
        buf = (seg + punct).strip()
        if punct:
            rebuilt.append(buf)
            buf = ""
    if buf:
        rebuilt.append(buf)
    return rebuilt

def ocr_image_pil(image_pil: Image.Image) -> str:
    ocr_result, _ = _ocr(image_pil)
    if not ocr_result:
        return ""
    return "".join([line[1] for line in ocr_result])

# ---------- 主函数 ----------
def scan_pdf_tables_and_text(pdf_path: str, dpi: int = DPI) -> tuple[List[PageText], List[TableCrop]]:
    # 渲染全页，逐页处理
    pages = convert_from_path(pdf_path, dpi=dpi)
    page_texts: List[PageText] = []
    table_crops: List[TableCrop] = []
    for idx, pil_img in enumerate(pages, start=1):
        bgr = pil_to_bgr(pil_img)
        # 1) 表格检测
        boxes = detect_tables_on_bgr(bgr)
        # 2) 裁表格到内存
        for (x1, y1, x2, y2) in boxes:
            crop = bgr[y1:y2, x1:x2]
            ok, buf = cv2.imencode(".png", crop)
            if ok:
                table_crops.append(TableCrop(page=idx, png_bytes=buf.tobytes()))
        # 3) 遮蔽表格后 OCR 文本
        masked_bgr = mask_tables_on_bgr(bgr, boxes, margin_px=2)
        masked_pil = Image.fromarray(cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB))
        text_raw = ocr_image_pil(masked_pil)
        lines = clean_and_merge_lines(text_raw)
        if lines:
            page_texts.append(PageText(page=idx, text="\n".join(lines)))
    return page_texts, table_crops
