# app/retrieval/rerank.py
"""统一的 rerank 客户端（当前实现 DashScope gte-rerank-v2）。"""
from __future__ import annotations
from typing import List, Dict, Any
from http import HTTPStatus
import os

from app.core.settings import get_settings

try:
    import dashscope
except Exception:  # pragma: no cover
    dashscope = None


class RerankClient:
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider.lower()
        self.model = model
        if self.provider == "dashscope":
            if dashscope is None:
                raise RuntimeError("dashscope SDK not installed; pip install dashscope")
            # 读取 Key
            ak = os.getenv("DASHSCOPE_API_KEY")
            if not ak:
                raise RuntimeError("DASHSCOPE_API_KEY not set")
            dashscope.api_key = ak
        else:
            raise ValueError(f"Unsupported rerank provider: {provider}")

    def rerank(self, query: str, documents: List[str], top_n: int, return_documents: bool = False) -> List[
        Dict[str, Any]]:
        """
        返回列表，每项包含：{"index": 原序号, "score": float, "document": 可选}
        注：DashScope 的 gte-rerank-v2 分数越大相关性越强，典型范围 [0, ~10]。
        """
        if not documents:
            return []
        if self.provider == "dashscope":
            resp = dashscope.TextReRank.call(
                model=self.model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
                return_documents=return_documents,
            )
            if resp.status_code != HTTPStatus.OK:
                raise RuntimeError(f"dashscope rerank error: {resp}")
            # resp.output = {"results": [{"index": i, "relevance_score": s, "document": str?}, ...]}
            out = []
            for item in resp.output.get("results", []):
                idx = int(item.get("index", 0))
                score = float(item.get("relevance_score", 0.0))
                rec: Dict[str, Any] = {"index": idx, "score": score}
                if return_documents:
                    rec["document"] = item.get("document")
                out.append(rec)
            return out
        raise ValueError("Unsupported provider")


_client: RerankClient | None = None


def get_rerank_client() -> RerankClient:
    global _client
    if _client is None:
        s = get_settings()
        _client = RerankClient(provider=s.RERANK_PROVIDER, model=s.RERANK_MODEL)
    return _client
