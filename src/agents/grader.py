"""评估 Agent：批改学生的答案，并给出错在哪。

## 它是整个闭环的「眼睛」

学 → 练 → **评** → 调。前面两步（导师讲、出题练）都是往外给东西，
只有这一步把信息**收回来**：学生对没对？错在哪？掌握度该升还是该降？

所以这个 Agent 的输出必须可靠：判错一次，画像就跟着错一次，
而画像是路径规划的依据——错判会一路传导到「下一步该学什么」。

## 判定的容错

模型的输出会被尽力解析成 ``{"correct": ..., "score": ..., "feedback": ...}``。
JSON 解析不出来时，退一步从文本里找「正确 / 错误」这类词；
连关键词都找不到就**报错**，而不是默默判成「错」——
默默判错会污染画像，而且事后完全查不出原因。
"""

import math

from src.agents.base import Agent
from src.utils.json_parse import extract_json_object
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROMPT_TEMPLATE = """你是一位严格但鼓励人的编程课老师，请批改下面这道题。

题目：{question}

参考答案：{reference}

学生的回答：{answer}

批改要求：
1. 判断回答是否正确时**看意思对不对，不要抠字眼**：
   用词不同但意思对，算正确；堆了一堆关键词但逻辑不通，算错误。
2. 给一个 0 到 1 的分数：完全正确是 1，完全不对是 0，部分对给中间值。
3. 给一句反馈：对了就说清楚**对在哪**；错了要点出**错在哪一步**，
   不要只说「再想想」或「基本正确」这种没有信息量的评语。
4. 只输出 JSON 对象，不要任何解释文字、不要 Markdown 围栏。

输出格式：
{{"correct": true, "score": 1.0, "feedback": "..."}}"""

# 从纯文本里认判定的关键词。**必须先查否定**——「不正确」里含「正确」，
# 顺序反了会把明确判错的答案读成正确。这类顺序 bug 极难通过阅读发现。
_NEGATIVE_WORDS = ("不正确", "不对", "错误", "❌", "✗", "incorrect", "false")
_POSITIVE_WORDS = ("正确", "答对", "✓", "✅", "correct", "true")


class GraderAgent(Agent):
    """批改答案。"""

    name = "grader"

    def grade(self, question: str, *, reference: str, answer: str) -> dict:
        """批改一道题。

        参数
        ----
        question:
            题干。
        reference:
            参考答案（出题时一并生成的）。
        answer:
            学生写的答案。

        返回
        ----
        ``{"correct": bool, "score": float, "feedback": str}``。

        抛出
        ----
        ValueError:
            题干或学生答案为空，或模型输出完全无法判定对错时。
        """
        question = question.strip()
        answer = answer.strip()
        if not question:
            raise ValueError("题干不能为空")
        if not answer:
            # 空答案直接判错不合理——可能是学生没提交，也可能是程序传错了参数。
            # 两种情况下都该报错让人看一眼，而不是静默记一次「答错」
            raise ValueError("学生答案不能为空")

        prompt = PROMPT_TEMPLATE.format(
            question=question,
            reference=reference.strip() or "（未提供参考答案，请按你的判断）",
            answer=answer,
        )
        text = self._ask(prompt)

        data = extract_json_object(text)
        correct = _to_bool(data.get("correct")) if data else None
        if correct is None:
            correct = _from_text(text)
        if correct is None:
            raise ValueError(f"无法判断这道题的对错。模型输出：{text[:200]}")

        score = _to_score(data.get("score"), correct)
        feedback = str(data.get("feedback", "")).strip() or text.strip()

        logger.info("批改完成 | correct=%s | score=%.2f", correct, score)
        return {"correct": correct, "score": score, "feedback": feedback}


def _to_bool(value: object) -> bool | None:
    """把模型给的「对错」值转换成布尔。认不出来返回 ``None``。

    模型可能给 ``true``、``"true"``、``"正确"``、``1``、``"yes"`` 中的任意一种。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # 数字按「有没有分数」理解：0 分算错，其余算对
        return value > 0
    if isinstance(value, str):
        text = value.strip().lower()
        if any(word in text for word in _NEGATIVE_WORDS):
            return False
        if any(word in text for word in _POSITIVE_WORDS):
            return True
    return None


def _from_text(text: str) -> bool | None:
    """JSON 解析不出来时的兜底：从整段文字里找「正确 / 错误」。

    先查否定词。原因见 ``_NEGATIVE_WORDS`` 上面的注释。
    """
    lowered = text.lower()

    # 只有当否定词出现得比肯定词更靠前时才判错，避免「答案正确，但……」
    # 这种带转折的评语被误判
    negative_at = _first_index(lowered, _NEGATIVE_WORDS)
    positive_at = _first_index(lowered, _POSITIVE_WORDS)

    if negative_at is None and positive_at is None:
        return None
    if negative_at is None:
        return True
    if positive_at is None:
        return False
    return positive_at < negative_at


def _first_index(text: str, words: tuple[str, ...]) -> int | None:
    """找出这些词里第一个出现的位置，都没出现返回 ``None``。"""
    found = [text.find(word) for word in words]
    hits = [index for index in found if index != -1]
    return min(hits) if hits else None


def _to_score(value: object, correct: bool) -> float:
    """整理分数：认不出就用对错推算，越界就夹到区间内。

    越界**不报错**而是夹紧——这是模型给的分，不是程序参数。
    真出现 1.2 分，夹成 1.0 比让整个学习流程中断合理得多。
    """
    fallback = 1.0 if correct else 0.0

    if isinstance(value, str):
        # 模型常把数字写成字符串，例如 "score": "0.5"，顺手认下来
        try:
            value = float(value.strip())
        except ValueError:
            return fallback

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    if math.isnan(value) or math.isinf(value):
        return fallback

    return round(min(max(float(value), 0.0), 1.0), 4)
