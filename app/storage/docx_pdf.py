# app/storage/docx_pdf.py
from __future__ import annotations
import os, tempfile, shutil
from pathlib import Path


def docx_to_pdf(input_path: str) -> str:
    """
    将 .doc/.docx 转为 .pdf，返回生成的 PDF 路径。
    - 优先使用 docx2pdf（Windows 下依赖已安装的 MS Word）
    - 失败则抛异常（后续你需要备用方案可再加）
    """
    from docx2pdf import convert  # 延迟导入，避免非 Windows 环境报错

    input_path = str(Path(input_path).resolve())
    lower_input = input_path.lower()
    if not (lower_input.endswith(".docx") or lower_input.endswith(".doc")):
        raise ValueError("docx_to_pdf only accepts .docx/.doc")

    tmpdir = tempfile.mkdtemp(prefix="docx2pdf_")
    out_pdf_path = Path(tmpdir) / (Path(input_path).stem + ".pdf")
    out_pdf = str(out_pdf_path)
    try:
        if lower_input.endswith(".docx"):
            # docx2pdf 支持 (in_file, out_file) 直接转换
            convert(input_path, out_pdf)
        else:
            _convert_doc_to_pdf(input_path, out_pdf)

        if not os.path.exists(out_pdf):
            raise RuntimeError("docx2pdf convert finished but output missing")
        return out_pdf
    except Exception as e:
        # 清理并上抛，方便上层 fallback（如果以后你要加备用方案）
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def _convert_doc_to_pdf(input_path: str, out_pdf: str) -> None:
    """使用 Windows Word COM 接口将 .doc 转成 PDF。"""

    input_abs = str(Path(input_path).resolve())
    out_pdf_abs = str(Path(out_pdf).resolve())

    try:
        import win32com.client as win32_client  # type: ignore
    except ImportError:
        win32_client = None

    word = None
    document = None
    try:
        if win32_client is not None:
            word = win32_client.DispatchEx("Word.Application")
        else:
            try:
                import comtypes.client
            except ImportError as exc:  # pragma: no cover - 平台相关
                raise RuntimeError(
                    "Converting .doc files requires pywin32 or comtypes"
                ) from exc
            word = comtypes.client.CreateObject("Word.Application")

        word.Visible = False
        word.DisplayAlerts = 0
        document = word.Documents.Open(input_abs)
        document.SaveAs(out_pdf_abs, FileFormat=17)  # 17 == wdFormatPDF
    finally:
        if document is not None:
            document.Close(False)
        if word is not None:
            word.Quit()
