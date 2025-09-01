"""
contracts_and_interfaces.py

阶段 1：领域模型（数据结构）+ 接口（函数/类签名）。
说明：函数体暂不实现，仅用中文 docstring 说明职责与输入输出，便于逐步填充。
Python 版本：3.11+
依赖：pydantic>=2, typing-extensions（若低版本 Python）
"""
from __future__ import annotations

from enum import Enum
from typing import List, Dict, Optional, Tuple, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


# =========================
# 枚举与类型别名
# =========================


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LOG_LEVEL: str = "INFO"
    DEBUG_PREVIEW_CHARS: int = 120
    DEBUG_SHOW_TOPN: int = 8
    RERANK_TOPN: int = 50
    # 其它原有字段也放这里（EMB_BASE_URL/KEY, CHROMA_PATH, RERANK_MODEL 等）


@lru_cache
def get_settings() -> Settings:
    return Settings()


class FileType(str, Enum):
    pdf_text = "pdf_text"
    pdf_scan = "pdf_scan"
    docx = "docx"
    xlsx = "xlsx"


class SegmentType(str, Enum):
    text = "text"
    table = "table"
    chart = "chart"
    figure = "figure"
    photo = "photo"
    excel_sheet = "excel_sheet"


class NonTextKind(str, Enum):
    table = "table"
    chart = "chart"
    figure = "figure"
    photo = "photo"


class JobStage(str, Enum):
    uploading = "UPLOADING"
    queued = "QUEUED"
    converting = "CONVERTING/ANALYZING"
    chunking = "CHUNKING"
    embedding = "EMBEDDING"
    indexing = "INDEXING"
    ready = "READY"
    failed = "FAILED"
    canceled = "CANCELED"


class QAStage(str, Enum):
    queued = "QUERY_QUEUED"
    retrieve = "RETRIEVE"
    rerank = "RERANK"
    generate = "GENERATE"
    done = "DONE"
    no_answer = "NO_ANSWER"
    failed = "FAILED"


DocID = str
FileID = str
TaskID = str
ChunkID = str
AnswerID = str
UserID = str


# =========================
# 配置（与文档 §8 对应）
# =========================

class Settings(BaseSettings):
    """系统运行时配置（可由环境变量覆盖）。"""

    # —— 嵌入服务（OpenAI 兼容 /embeddings） ——
    EMB_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 例如："https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMB_API_KEY: str = os.getenv("DASHSCOPE_API_KEY")  # 从环境变量读取，别写死
    print(EMB_API_KEY)
    EMBEDDING_MODEL: str = "text-embedding-v4"  # 你原来已有就沿用

    # —— 日志配置 ——
    LOG_LEVEL: str = "DEBUG"
    DEBUG_PREVIEW_CHARS: int = 120  # 日志里每条文本最多打印这么多字符
    DEBUG_SHOW_TOPN: int = 4  # rerank/最终排名等每次最多展示多少条

    # —— Chroma 存储路径 ——
    CHROMA_PATH: str = "./data/chroma"

    # —— rerank模型 ——
    RERANK_PROVIDER: str = "dashscope"
    RERANK_MODEL: str = "gte-rerank-v2"
    RERANK_TOPN: int = 50  # 统一送入 reranker 的候选上限

    # —— 召回增强 ——
    NEIGHBOR_WINDOW_PAGES: int = 1
    NEIGHBOR_PER_SIDE: int = 1
    APPENDIX_TAIL_RATIO: float = 0.10
    APPENDIX_TOP: int = 2

    # —— 标题索引 ——
    ENABLE_TITLE_INDEX: bool = True
    TITLE_TOPK: int = 3

    # —— 重排权重 ——
    W_SIM: float = 0.60
    W_REWRITE: float = 0.10
    W_TITLE: float = 0.08
    W_NEIGHBOR: float = 0.08
    W_APPENDIX: float = 0.06
    W_CLASS: float = 0.08
    CLASS_BONUS_table: float = 0.06
    CLASS_BONUS_chart: float = 0.03
    CLASS_BONUS_figure: float = 0.00
    CLASS_BONUS_photo: float = 0.00
    MIN_SIM: Optional[float] = None  # 训练后设定

    # —— 队列/轮询 ——
    QUEUE_BACKEND: Literal["rq", "celery"] = "rq"
    WORKERS_convert: int = 2
    WORKERS_ocr: int = 4
    WORKERS_chunk: int = 2
    WORKERS_embed: int = 4
    WORKERS_index: int = 2
    WORKERS_qa: int = 2

    POLL_INITIAL_MS: int = 1000
    POLL_MAX_MS: int = 3000
    POLL_BACKOFF: float = 1.0

    # —— 缓存 ——
    ENABLE_IMG_CACHE: bool = True
    IMG_PHASH_HAMMING_MAX: int = 6
    ENABLE_EMB_CACHE: bool = True
    EMB_CACHE_BACKEND: Literal["redis", "sqlite"] = "redis"
    CACHE_TTL_DAYS: int = 30

    # —— 限额 ——
    MAX_FILE_SIZE_MB: int = 50
    MAX_PAGES_PER_FILE: int = 300
    MAX_IMAGES_PER_FILE: int = 800
    MAX_FILES_PER_BATCH: int = 20
    MAX_CONCURRENT_TASKS_PER_USER: int = 3
    ANSWER_FINAL_K: int = 4

    # —— ETA ——
    ETA_WINDOW_MINUTES: int = 30
    ETA_MIN_SAMPLE: int = 50

    # —— 审计与指标 ——
    AUDIT_LOG_PATH: str = "./logs/audit.jsonl"
    PROMETHEUS_ENABLE: bool = True

    # —— 模型选择 ——
    LLM_CHAT_MODEL: str = "qwen-turbo"  # 示例；实际以你接的提供商为准


# =========================
# 核心数据模型
# =========================

class FileMeta(BaseModel):
    """上传文件的元信息与判定结果。"""
    file_id: FileID
    doc_id: DocID
    owner_id: UserID
    filename: str
    mime: str
    size_bytes: int
    file_type: FileType
    page_count: Optional[int] = None
    image_count: Optional[int] = None
    created_at: datetime


class PageMeta(BaseModel):
    """单页物理与判定信息（供 OCR/切片阶段使用）。"""
    doc_id: DocID
    page_number: int
    width: int
    height: int
    dpi: Optional[int] = None
    is_scanned: bool = False


class NeighborInfo(BaseModel):
    prev_page: Optional[int] = None
    next_page: Optional[int] = None
    prev_chunk_id: Optional[ChunkID] = None
    next_chunk_id: Optional[ChunkID] = None


class ChunkMeta(BaseModel):
    chunk_id: ChunkID
    doc_id: DocID
    segment_type: SegmentType
    page_from: Optional[int] = None
    page_to: Optional[int] = None
    title_guess: Optional[str] = ""
    page_estimated: bool = False
    appendix_flag: bool = False
    neighbors: Optional[NeighborInfo] = None
    # —— 扁平化的邻近信息（用于检索/分析；全部是“原子类型或 None”）——
    neighbors_prev_page: Optional[int] = None
    neighbors_next_page: Optional[int] = None
    neighbors_prev_chunk_id: Optional[str] = None
    neighbors_next_chunk_id: Optional[str] = None


class Chunk(BaseModel):
    """用于向量化与检索的最小单元。文本或来自非文字图片的摘要/JSON。"""
    chunk_id: ChunkID
    text: str  # 文本切片正文或非文字摘要；若是表格，可存 JSON 的紧凑字符串版本
    metadata: ChunkMeta


class TablePayload(BaseModel):
    headers: List[str] = []
    rows: List[List[str]] = []
    notes: List[str] = []


class NonTextExtraction(BaseModel):
    """非文字图片经 LLM 判类后的结构化结果。可被转为 Chunk。"""
    kind: NonTextKind
    title: str = ""
    text: str = ""
    table: Optional[TablePayload] = None


class TaskProgress(BaseModel):
    task_id: TaskID
    stage: JobStage
    phase_percent: int = 0
    eta_seconds: Optional[int] = None
    queue_len: Optional[int] = None
    current_stage: Optional[str] = None
    error: Optional[str] = None


class QARequest(BaseModel):
    owner_id: UserID
    query: str
    top_k: int = 8
    session_id: Optional[str] = None


class Citation(BaseModel):
    doc_id: DocID
    chunk_id: ChunkID
    filename: str
    page_hint: Optional[str] = None  # 例："p.12" 或 "Sheet: 考核结果"


class QAAnswer(BaseModel):
    answer_id: AnswerID
    stage: QAStage
    text: Optional[str] = None
    citations: List[Citation] = []
    no_answer_reason: Optional[str] = None
    final_k: int | None = None


class CandidateFeatures(BaseModel):
    sim: float = 0.0
    rewrite: float = 1.0
    title: int = 0
    neighbor: int = 0
    appendix: int = 0
    class_bonus: float = 0.0


class RetrieveCandidate(BaseModel):
    chunk: Chunk
    channel: List[str]  # ["orig"|"rewrite"|"title"|"neighbor"|"appendix"|"nontext2"]
    features: CandidateFeatures


class ScoredCandidate(BaseModel):
    candidate: RetrieveCandidate
    score: float


class AuditLogEntry(BaseModel):
    ts: datetime
    request_id: str
    user_id: UserID
    query: str
    rewrites: List[str]
    candidates: List[Dict]  # 简化：落盘时存扁平字典
    final_top: List[ChunkID]
    no_answer_reason: Optional[str] = None
    dur_ms: Dict[str, int] = {}


# =========================
# 接口：服务层（无实现，仅职责说明）
# =========================

class IngestService:
    """处理上传文件：入队、去重、分发到各 Worker 阶段。"""

    def schedule_ingest(self, owner_id: UserID, filename: str, mime: str, file_bytes: bytes) -> Tuple[FileID, TaskID]:
        """接收文件，计算 doc_id 去重；创建文件记录并投递 convert 任务。返回 file_id 与 task_id。"""
        raise NotImplementedError

    def get_task_progress(self, task_id: TaskID) -> TaskProgress:
        """查询任务进度，聚合各阶段完成度，估算 ETA 与排队数。"""
        raise NotImplementedError


class QAService:
    """问答编排：改写、多路召回、重排、生成与追踪。"""

    def submit_query(self, owner_id: UserID, query: str, top_k: int = 8, session_id: Optional[str] = None) -> AnswerID:
        """创建 answer 记录并入队 QA 流程（或同步执行检索/重排、异步生成）。返回 answer_id。"""
        raise NotImplementedError

    def get_answer(self, answer_id: AnswerID) -> QAAnswer:
        """返回当前问答阶段状态与已生成的部分内容/证据。NO_ANSWER 时提供原因。"""
        raise NotImplementedError


# =========================
# 接口：检索/重排/拼接（纯函数签名）
# =========================

def generate_rewrites(query: str, k: int = 4) -> List[str]:
    """生成 K 条改写，使用 LLM；返回 [rewrite1, …]。"""
    raise NotImplementedError


def retrieve_candidates(query_or_rewrite: str, top_k: int, title_index: bool = False) -> List[RetrieveCandidate]:
    """
    在主内容索引或标题索引中检索候选；
    - 当 title_index=True 时，命中文件/Sheet 后需将其主内容切片转为候选并打上 title 特征。
    """
    raise NotImplementedError


def nontext_second_pass(doc_ids: List[DocID], strong_rewrite: str, top_k: int = 2) -> List[RetrieveCandidate]:
    """在 {table,chart,figure,photo} 过滤集上做二次召回。"""
    raise NotImplementedError


def neighbor_force_recall(text_candidates: List[RetrieveCandidate]) -> List[RetrieveCandidate]:
    """基于候选的邻近信息强制加入同页/前后页的图表/表格。"""
    raise NotImplementedError


def appendix_force_recall(doc_ids: List[DocID]) -> List[RetrieveCandidate]:
    """对每个文档追加附表区 Top-M 表格候选。"""
    raise NotImplementedError


def rerank(candidates: List[RetrieveCandidate], settings: Settings) -> List[ScoredCandidate]:
    """根据 §6.2 的加权公式计算 score 排序，必要时判定无答案。"""
    raise NotImplementedError


def build_final_context(scored: List[ScoredCandidate], max_tokens: int = 7000) -> str:
    """拼接 Top-N 文本切片，并为每个文本切片最多追加 1 个图表/表格证据。"""
    raise NotImplementedError


# =========================
# 接口：缓存与哈希
# =========================

def cache_get_or_set_img(model: str, image_bytes: bytes) -> NonTextExtraction:
    """
    非文字图片哈希缓存：
    - 使用 sha256 与感知哈希 pHash 组合键；
    - 命中则直接返回历史 LLM 结果；未命中则调用 LLM 并写回缓存。
    """
    raise NotImplementedError


def cache_get_or_set_embedding(model: str, text: str) -> List[float]:
    """
    嵌入哈希缓存：
    - key = sha256(normalize(text))；命中返回向量；未命中调用模型并写回。
    """
    raise NotImplementedError


# =========================
# 接口：限额与 ETA
# =========================

class LimitCheck(BaseModel):
    ok: bool
    reason: Optional[str] = None
    hit_flags: List[str] = []


def check_limits(meta: FileMeta, settings: Settings) -> LimitCheck:
    """校验大小/页数/图片数/并发任务数等限额，返回是否允许继续处理。"""
    raise NotImplementedError


def estimate_eta(task_id: TaskID) -> int:
    """基于队列的滑动窗口统计估算 ETA（秒），用于前端展示。"""
    raise NotImplementedError


# =========================
# 接口：审计日志
# =========================

def write_audit_log(entry: AuditLogEntry) -> None:
    """将一次问答的关键调参信息写入 JSONLines，便于后续分析与复现。"""
    raise NotImplementedError
