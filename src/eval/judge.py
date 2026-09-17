"""要点判定的评估器：让大模型判断「这段回答里有没有提到这几个要点」。

## 为什么不用关键词匹配

``metrics.match_keywords`` 已经有一个零成本的做法，但它在实验一里不够用：

    要点写着「指出薄弱点在递归的终止条件」
    回答写着「你递归的出口条件还得再练练」

「终止条件」和「出口条件」是同一个意思，关键词匹配认不出来，会判成没覆盖。
实验一要区分的是「模型知不知道这个学生的情况」，而不是「它有没有用一样的词」，
所以必须让模型来判。

## 一次判完所有要点

每个要点单独调一次模型是最直观的写法，但调用次数会翻好几倍
（3 个条件 × 3 个问题 × 3 个要点 = 27 次）。把要点列成清单让它一次判完，
调用次数降到 9 次，结果也更容易保持一致——同一个回答的几个要点
由同一次判断给出，不会出现「第一个要点判得松、第三个判得严」。
"""

import json
from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel

from src.llm import create_chat_model
from src.utils.json_parse import extract_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROMPT_TEMPLATE = """下面是一段回答，以及若干个要点。请逐条判断每个要点是否**在这段回答里被表达过**。

判定标准：
1. **看意思，不看用词**。回答用别的说法表达了同样的意思，算「有」；
2. 只是沾边、说得含糊、或者需要读者自己脑补的，算「没有」；
3. 逐条独立判断，不要因为前面判过就影响后面。

学生的问题：{question}

回答：
{answer}

要点清单：
{points}

只输出一个 JSON 数组，元素个数必须正好是 {count} 个，每项为 true 或 false。
不要任何解释文字、不要 Markdown 围栏。
例如：[true, false, true]"""


def judge_points(
    question: str,
    answer: str,
    points: Sequence[str],
    *,
    model: BaseChatModel | None = None,
) -> list[bool]:
    """逐条判断要点是否被回答覆盖。

    参数
    ----
    question:
        学生问的问题（给判定者当参照，避免断章取义）。
    answer:
        待判定的回答。
    points:
        要点清单。
    model:
        聊天模型。不传用默认 DeepSeek；测试时塞假模型即可离线跑。

    返回
    ----
    布尔列表，长度与 ``points`` 一致。

    模型给的数量不对时**按缺失位置补 False**，而不是抛错：
    评测脚本跑到一半崩掉、还得从头再花钱跑一遍，代价比少判一条大得多。
    数量对不上会记一条警告，事后能查出来。
    """
    if not points:
        return []

    model = model or create_chat_model()
    numbered = "\n".join(f"{index}. {point}" for index, point in enumerate(points, start=1))

    prompt = PROMPT_TEMPLATE.format(
        question=question.strip() or "（未提供）",
        answer=answer.strip(),
        points=numbered,
        count=len(points),
    )

    reply = model.invoke(prompt)
    text = reply.content if hasattr(reply, "content") else str(reply)

    marks = _parse_marks(text, len(points))
    if len(marks) != len(points):
        logger.warning(
            "判定结果数量对不上 | 期望 %d 条，实际 %d 条 | 原始输出：%s",
            len(points),
            len(marks),
            text[:120],
        )

    # 补齐或截断，保证长度一致——上层可以直接和要点列表对齐使用
    marks = (marks + [False] * len(points))[: len(points)]
    return marks


def _parse_marks(text: str, expected: int) -> list[bool]:
    """从模型输出里挖出布尔数组。

    模型有时写成 ``[true, false]``，有时写成 ``[1, 0]``，
    还有写成 ``["是", "否"]`` 的，都得认。
    """
    data = extract_json(text)
    if not isinstance(data, list):
        return []

    marks: list[bool] = []
    for item in data:
        value = _to_bool(item)
        if value is not None:
            marks.append(value)
    return marks[:expected] if len(marks) >= expected else marks


def _to_bool(value: object) -> bool | None:
    """把各种写法折算成布尔。认不出来返回 ``None``（这一条不计入）。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "是", "有", "yes", "y", "1", "覆盖", "命中"}:
            return True
        if text in {"false", "否", "没有", "no", "n", "0", "未覆盖", "未命中"}:
            return False
    return None


def to_json(data: object) -> str:
    """调试用：把结果打成 JSON 字符串。"""
    return json.dumps(data, ensure_ascii=False, indent=2)
