from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ingest_service import ingest_service, _DOC_TO_FILE, DOC_CHUNKS, DOC_TO_FILE  # 仅演示用读映射
from app.domain.contracts import TaskProgress, FileMeta, Chunk
from typing import List, Optional

router = APIRouter(prefix="/v1", tags=["files"])


@router.post("/files")
async def upload_file(file: UploadFile = File(...)):
    # 演示：读取字节；owner 固定 demo
    # try:
    data = await file.read()
    file_id, task_id = ingest_service.schedule_ingest(
        owner_id="demo_user",
        filename=file.filename,
        mime=file.content_type or "",
        file_bytes=data
    )
    # 查回 doc_id（演示：从映射表获取，正式应由 service 返回或查询 FileMeta）
    # 这里读取一次没问题；后面我们会把 FileMeta 查询接口补上
    for doc_id, fid in _DOC_TO_FILE.items():
        if fid == file_id:
            return {"file_id": file_id, "task_id": task_id, "doc_id": doc_id}
    return {"file_id": file_id, "task_id": task_id}
    # except ValueError as e:
    #     # 由 check_limits 抛出，转成 400
    #     raise HTTPException(status_code=400, detail=str(e))


@router.get("/tasks/{task_id}", response_model=TaskProgress)
async def get_task(task_id: str):
    return ingest_service.get_task_progress(task_id)


# —— 新增：查询文件元信息 ——
@router.get("/files/{file_id}", response_model=FileMeta)
async def get_file_meta(file_id: str):
    meta = ingest_service.get_file_meta(file_id)
    if not meta:
        raise HTTPException(status_code=404, detail="file_not_found")
    return meta


@router.get("/files/{file_id}/chunks", response_model=List[Chunk])
async def preview_chunks(file_id: str, limit: Optional[int] = 5):
    # 通过 file_id 找 doc_id
    doc_id = None
    for d, fid in DOC_TO_FILE.items():
        if fid == file_id:
            doc_id = d
            break
    if not doc_id:
        raise HTTPException(status_code=404, detail="file_not_found")

    chunks = DOC_CHUNKS.get(doc_id, [])
    if limit is not None:
        chunks = chunks[: int(limit)]
    return chunks
