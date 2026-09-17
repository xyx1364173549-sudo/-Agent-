"""从大模型的自由文本里挖出 JSON。

这是一个**反复出现**的需求，所以单独放一个文件：

    拆解目标   要 ``[{"topic": ..., "depends_on": [...]}]``
    生成题目   要 ``[{"question": ..., "answer": ...}]``
    批改答案   要 ``{"correct": true, "feedback": "..."}``

而模型的输出往往长这样：

    ```json
    [{"topic": "递归"}]
    ```

或者：

    好的，这是结果：{"correct": true} 希望对你有帮助。

再或者干脆少一个引号、多一个逗号。所以需要一个够皮实的解析器。
"""

import json
import re
from typing import Any

# 匹配 Markdown 代码围栏：```json ... ``` 或 ``` ... ```
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def strip_fence(text: str) -> str:
    """去掉 Markdown 代码围栏。没有围栏就原样返回。"""
    match = _FENCE.search(text)
    return match.group(1).strip() if match else text.strip()


def extract_json(text: str) -> Any | None:
    """从一段文字里找出第一个能解析成功的 JSON，找不着返回 ``None``。

    做法是**从每个候选起点试一次**：只要看到 ``{`` 或 ``[``，就从那里开始
    解析，成功了就返回，失败了就换下一个起点继续试。

    比「取第一个 ``[`` 到最后一个 ``]``」的做法稳，因为后者在文本里出现多个
    JSON 片段时（例如模型先给个示例又给正式答案）会把两段黏在一起，
    解析必然失败。``raw_decode`` 则只吃它能吃下的那一段，剩下的留着不管。
    """
    if not text:
        return None

    cleaned = strip_fence(text)

    # 大多数情况下整个输出就是 JSON，先直接试最快的路
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        return value

    return None


def extract_json_array(text: str) -> list:
    """只要 JSON 数组，拿不到就返回空列表。"""
    value = extract_json(text)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    # 模型有时会包一层，例如 {"topics": [...]}，把里面的数组翻出来
    if isinstance(value, dict):
        for inner in value.values():
            if isinstance(inner, list):
                return [item for item in inner if isinstance(item, dict)]
    return []


def extract_json_object(text: str) -> dict:
    """只要 JSON 对象，拿不到就返回空字典。"""
    value = extract_json(text)
    if isinstance(value, dict):
        return value
    # 反过来：模型把对象包在数组里，例如 [{...}]
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return item
    return {}
