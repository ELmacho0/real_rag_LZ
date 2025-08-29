from functools import lru_cache
from app.domain.contracts import Settings


@lru_cache()
def get_settings() -> Settings:
    # 支持从环境变量覆盖（pydantic-settings 默认行为）
    return Settings()
