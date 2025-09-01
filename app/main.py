from fastapi import FastAPI
from app.api.routes_files import router as files_router
from app.api.routes_query import router as query_router
from app.core.logging import setup_logging

app = FastAPI(title="RAG Demo API", version="v1")

app.include_router(files_router)
app.include_router(query_router)

setup_logging()


@app.get("/health")
def health():
    return {"ok": True}
