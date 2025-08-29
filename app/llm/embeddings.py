# app/llm/embeddings.py
from __future__ import annotations
from typing import List
import requests

from app.core.settings import get_settings


class EmbeddingClient:
    """极简的 OpenAI 兼容 /embeddings 客户端。"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60, dim: int = 2048) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.dim = dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        batch_size: int = 5
        out= []
        for i in range(0, len(texts), batch_size):
            url = f"{self.base_url.rstrip('/')}/embeddings"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "input": texts[i:i+batch_size],
                # 可选，若需降维或指定返回格式，保持与 SDK 一致：
                "dimensions": self.dim,           # 例如 1024
                "encoding_format": "float",
            }
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            r.raise_for_status()
            obj = r.json()
            out.extend(item["embedding"] for item in obj.get("data", []))
        print("已完成",len(texts),"条chrunck向量化")

        return out


# 全局单例（按需初始化）
_client: EmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient:
    global _client
    if _client is None:
        s = get_settings()
        print("尝试初始化模型")
        if not s.EMB_BASE_URL or not s.EMB_API_KEY:
            raise RuntimeError("Embedding service not configured: set EMB_BASE_URL & EMB_API_KEY")
        _client = EmbeddingClient(base_url=s.EMB_BASE_URL, api_key=s.EMB_API_KEY, model=s.EMBEDDING_MODEL)
        print("初始化模型成功")
    return _client
