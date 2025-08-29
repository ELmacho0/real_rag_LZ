# app/retrieval/rewrite.py
from __future__ import annotations
from typing import List


def generate_rewrites(q: str, k: int = 4) -> List[str]:
    q = (q or "").strip()
    if not q:
        return []
    out = []
    # 简单模板改写（占位）：同义词/换序/补全
    out.append(q)
    out.append(q.replace("谁", "哪位"))
    out.append(f"请根据文档说明：{q}")
    out.append(q + "？" if not q.endswith("？") else q[:-1])
    out.append(q.replace("审批", "审核"))
    # 去重、截断
    dedup = []
    seen = set()
    for s in out:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup[: max(1, k)]
