import hashlib, os
from pathlib import Path
from typing import Tuple
from app.domain.contracts import FileType

DATA_ROOT = Path("./data").resolve()


def compute_doc_id(file_bytes: bytes) -> str:
    """对原始字节做 sha256，作为 doc_id。"""
    return hashlib.sha256(file_bytes).hexdigest()


def guess_file_type(filename: str, mime: str | None) -> FileType:
    """简单按扩展名判断（先不区分 pdf_text / pdf_scan，后续接 PyMuPDF/OCR 再细分）。"""
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return FileType.pdf_text
    if name.endswith(".docx") or name.endswith(".doc"):
        return FileType.docx
    if name.endswith(".xlsx"):
        return FileType.xlsx
    # 默认按 pdf_text 兜底
    return FileType.pdf_text


def save_original(owner_id: str, doc_id: str, filename: str, file_bytes: bytes) -> Path:
    """将原始文件落盘，返回保存路径。"""
    ext = Path(filename).suffix or ".bin"
    folder = DATA_ROOT / "original" / owner_id / doc_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"original{ext}"
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path
