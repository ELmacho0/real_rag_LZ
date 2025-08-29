# app/retrieval/fusion.py（替换原文件）
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Cand:
    chunk_id: str
    doc_id: str
    text: str
    page_from: int | None
    # 记录（不参与最终相关性，仅用于日志/兜底）
    emb_sim: float | None = None     # 向量相似度（可无）
    # rerank 分数（核心相关性来源）
    rel: float = 0.0                 # gte-rerank-v2 的 relevance_score
    # 命中特征
    title_hit: bool = False
    neighbor_boost: bool = False
    appendix_boost: bool = False
    rewrite_weight: float = 1.0      # Q=1.0，改写=0.8，标题/强制=0.6
    class_bonus: float = 0.0         # table=+0.06, chart=+0.03
    multi_hit: int = 1               # 同一 chunk 被多路命中次数
    # 附带信息
    filename: str | None = None

# —— 权重（可移 Settings） ——
W_REL = 0.60       # rerank 相关性
W_REWRITE = 0.10
W_TITLE = 0.08
W_NEIGHBOR = 0.08
W_APPENDIX = 0.06
W_CLASS = 0.08
W_MULTI = 0.03     # 多路命中小加成（每 +1 命中加 0.03，上限自行把控）

MIN_REL = 0.25     # 无答案阈值（基于 rerank 分数尺度，经验值，可调）


def score(c: Cand) -> float:
    return (
        W_REL * max(0.0, c.rel)
        + W_REWRITE * c.rewrite_weight
        + W_TITLE * (1.0 if c.title_hit else 0.0)
        + W_NEIGHBOR * (1.0 if c.neighbor_boost else 0.0)
        + W_APPENDIX * (1.0 if c.appendix_boost else 0.0)
        + W_CLASS * c.class_bonus
        + W_MULTI * max(0, c.multi_hit - 1)
    )


def is_unreliable_top(c: Cand) -> bool:
    no_boost = (not c.title_hit) and (not c.neighbor_boost)
    return (c.rel < MIN_REL) and no_boost