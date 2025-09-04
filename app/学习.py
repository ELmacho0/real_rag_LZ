import os
import base64
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# 本地图片转 Base64
def image_to_base64(path):
    with open(path, "rb") as f:
        return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"


# 主逻辑
local_image_path = "D:\智能知识库\PDF转换输出\wechat_2025-09-02_223658_506.png"
image_url = image_to_base64(local_image_path)
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
    {"kind": "table|chart|figure|photo",\n "title": "<若能识别/推断的标题>",\n  "text": "<单纯的表格的JSON，或其他内容的详细描述>"\n}\n"""
    "要求：严格按照我的输出格式输出，不要输出其他内容！"
)
completion = client.chat.completions.create(
    model="qwen-vl-max-2025-08-13",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": PROMPT}
        ]
    }],
    # 新增参数设置
    temperature=0.1,  # 控制输出随机性（0-1），值越低输出越稳定准确
    top_p=0.9,  # 核采样概率阈值，与temperature共同控制输出质量
    max_tokens=5000,  # 最大输出长度（对表格结构化输出尤为重要）
    presence_penalty=1.0,  # 鼓励生成新信息，避免重复内容
    frequency_penalty=-1,  # 抑制高频词重复，提升描述多样性
)
a = completion.model_dump_json()
print(completion.model_dump_json())

"""
候选：
qwen-vl-plus-2025-08-15
qwen-vl-max-2025-08-13
"""
