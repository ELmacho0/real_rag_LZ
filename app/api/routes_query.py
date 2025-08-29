from fastapi import APIRouter
from app.services.qa_service import qa_service
from app.domain.contracts import QARequest, QAAnswer

router = APIRouter(prefix="/v1", tags=["query"])


@router.post("/query")
async def submit_query(req: QARequest):
    ans_id = qa_service.submit_query(owner_id=req.owner_id, query=req.query, top_k=req.top_k, session_id=req.session_id)
    return {"answer_id": ans_id}


@router.get("/answers/{answer_id}", response_model=QAAnswer)
async def get_answer(answer_id: str):
    return qa_service.get_answer(answer_id)
