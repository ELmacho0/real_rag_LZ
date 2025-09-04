# app/llm/vision.py
from __future__ import annotations
import base64, json, os
from typing import Dict, Any
from openai import OpenAI
import re

from app.core.settings import get_settings

PROMPT = (
    """你将接收一张来自文档的图片，请先判断它属于哪一类：\n
    - "table"（可用行列表达的结构化数据）\n
    - "chart"（折线/柱状/饼/面积等可视化）\n
    - "figure"（流程图/架构图/示意图/截图）\n
    - "photo"（照片/插画）\n
    - 若不确定，选最接近的一类。\n\n
    - 识别图片的标题，如果没有识别到标题，则根据内容猜测一个标题\n\n
    - 如果判断判断是table，则将该表格结构化成JSON格式；如果是其他，则转化成对其内容的详细描述\n
    输出格式如下：\n
    {"kind": "table|chart|figure",\n "title": "<识别/推断的标题>",\n  "text": "<单纯的表格的JSON，或其他内容的详细描述>"\n}\n"""
    "要求：严格按照我的输出格式输出，不要输出其他内容！"
)
_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        s = get_settings()
        _client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY", ""), base_url=s.VISION_BASE_URL)
    return _client


def image_bytes_to_data_url(b: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(b).decode()


def analyze_image(image_bytes: bytes, model: str | None = None) -> Dict[str, Any]:
    """调用多模态模型，返回规范化的 dict。"""
    s = get_settings()
    cli = get_client()
    model = model or s.VISION_MODEL
    data_url = image_bytes_to_data_url(image_bytes)
    resp = cli.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": PROMPT}
            ]
        }],
        temperature=0.1,  # 控制输出随机性（0-1），值越低输出越稳定准确
        top_p=0.9,  # 核采样概率阈值，与temperature共同控制输出质量
        max_tokens=5000,  # 最大输出长度（对表格结构化输出尤为重要）
        presence_penalty=1.0,  # 鼓励生成新信息，避免重复内容
        frequency_penalty=-1,  # 抑制高频词重复，提升描述多样性
    )
    text = resp.choices[0].message.content or "{}"
    try:
        text = text[text.find("{"):text.rfind("}") + 1] if "{" in text and "}" in text else text
        obj = json.loads(text)
    except Exception:
        obj = {"kind": "figure", "title": "", "text": text}
    # 兜底字段
    obj.setdefault("kind", "figure")
    obj.setdefault("title", "")
    obj.setdefault("text", "")
    return obj


def table_json_to_text(dirty_str: str) -> str:
    # 将字符串中的换行符、反斜杠和空格去掉
    cleaned_str = re.sub(r'[\n\\ ]', '', dirty_str)
    return cleaned_str
