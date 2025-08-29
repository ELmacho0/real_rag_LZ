from typing import List
from app.domain.contracts import FileMeta, Settings
from app.domain.contracts import LimitCheck  # 已在 contracts 里定义签名


def check_limits(meta: FileMeta, settings: Settings) -> LimitCheck:
    """先做最基本的大小校验；页数/图片数等等我们后面有了解析能力再补。"""
    hits: List[str] = []
    if meta.size_bytes > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        hits.append("file_size_exceeded")
        return LimitCheck(ok=False, reason="file_too_large", hit_flags=hits)
    return LimitCheck(ok=True, reason=None, hit_flags=hits)
