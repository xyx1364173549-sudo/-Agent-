"""出题 Agent：给一个知识点生成配套练习。

## 为什么要求模型输出 JSON

练习题不是给人看的文字，而是**要被程序继续处理的**：题目送去给学生做、
参考答案留下来批改用。所以它得是结构化的，不能是一大段 Markdown。

解析失败的风险是真实存在的——模型偶尔会把 JSON 写坏，或者裹一段解释。
所以这里用 ``src/utils/json_parse.py`` 那个皮实的解析器，并且**逐项校验**：
题干为空、答案缺失的条目直接丢掉，宁可少一道题，也不能出一道没法批改的题。
"""

from src.agents.base import Agent
from src.utils.json_parse import extract_json_array
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 一次最多出几道。太多学生做不完，也浪费 token。
DEFAULT_COUNT = 3

PROMPT_TEMPLATE = """你是一位编程课老师，请为下面的知识点出 {count} 道练习题。

知识点：{topic}

{focus}

出题要求：
1. 题目要**能检验是否真的理解**，不要出「什么是XX」这种背诵题；
2. 由易到难排列；
3. 每道题都要给出**参考答案**，答案要简短明确；
4. 只输出 JSON 数组，不要任何解释文字、不要 Markdown 围栏。

输出格式：
[
  {{"question": "题目", "answer": "参考答案", "hint": "给学生的一句提示"}}
]"""

# 学生有薄弱记录时追加的针对性要求
FOCUS_TEMPLATE = """这位学生的历史情况：
{context}

请针对他的薄弱之处出题，不要出他已经掌握的内容。"""


class QuizAgent(Agent):
    """生成练习题。"""

    name = "quiz"

    def generate(
        self,
        topic: str,
        *,
        count: int = DEFAULT_COUNT,
        context: str = "",
    ) -> list[dict[str, str]]:
        """为某个知识点出题。

        参数
        ----
        topic:
            知识点，例如「递归的终止条件」。
        count:
            出几道题。
        context:
            学生情况简报（可选）。传了就让题目更针对他的薄弱点——
            这正是「个性化」和「题库随机抽题」的区别。

        返回
        ----
        列表，每项形如 ``{"question": ..., "answer": ..., "hint": ...}``。

        抛出
        ----
        ValueError:
            一道有效题目都解析不出来时。不能返回空列表——
            上层会以为「出好了，做吧」，然后拿着空题单去问学生。
        """
        topic = topic.strip()
        if not topic:
            raise ValueError("知识点不能为空")
        if count < 1:
            raise ValueError(f"题目数量必须至少为 1，实际为 {count}")

        focus = FOCUS_TEMPLATE.format(context=context.strip()) if context.strip() else ""
        prompt = PROMPT_TEMPLATE.format(topic=topic, count=count, focus=focus)
        text = self._ask(prompt)

        questions = _clean(extract_json_array(text), limit=count)
        if not questions:
            raise ValueError(f"没能从模型输出里解析出题目。原始输出：{text[:200]}")

        logger.info("出题完成 | topic=%s | %d 道", topic, len(questions))
        return questions


def _clean(raw: list[dict], *, limit: int) -> list[dict[str, str]]:
    """校验并整理题目：题干和答案缺一不可，其余字段给默认值。"""
    questions: list[dict[str, str]] = []

    for item in raw:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            # 没有参考答案的题批改不了，直接丢。少一道题远好过一道没法批的题
            logger.warning("丢弃一道缺少题干或答案的题目")
            continue

        questions.append(
            {
                "question": question,
                "answer": answer,
                "hint": str(item.get("hint", "")).strip(),
            }
        )
        if len(questions) >= limit:
            break

    return questions
