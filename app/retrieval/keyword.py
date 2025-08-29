# app/retrieval/keyword.py
"""
关键词检索的最小工具：
- tokenize：极简分词（后续可替换为 jieba/THULAC/自研）
- score_text：朴素匹配打分（后续会被向量+重排替代）
"""
from __future__ import annotations
import re
from typing import List


def tokenize(q: str) -> List[str]:
    q = (q or "").lower()
    tokens = re.split(r"[^a-z0-9\u4e00-\u9fff]+", q)
    return [t for t in tokens if t]


def score_text(text: str, tokens: List[str]) -> int:
    if not text or not tokens:
        return 0
    t = text.lower()
    return sum(t.count(tok) for tok in tokens)
