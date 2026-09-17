"""三类功能 Agent 的测试：导师、出题、评估。

全部用假模型，不联网、不花钱、结果可复现。测试重心放在**模型的输出不听话时**
会怎样——真实运行时模型经常会：

    把 JSON 裹在代码围栏里
    多写两句解释
    把 true 写成「正确」
    少给一个字段
    给个越界的分数

这些都得被稳妥处理，判不出来时报错也比默默判错强。
"""

from __future__ import annotations

import json

import pytest

from src.agents import GraderAgent, QuizAgent, TutorAgent


class FakeReply:
    """假模型返回的消息对象，只要有个 content 属性就够。"""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    """假模型：invoke 时固定返回预先设定好的内容。"""

    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeReply:
        self.prompts.append(prompt)
        return FakeReply(self.content)


def json_text(data: object) -> str:
    return json.dumps(data, ensure_ascii=False)


# --------------------------------------------------------------------------
# 导师 Agent
# --------------------------------------------------------------------------


def test_tutor_returns_explanation() -> None:
    model = FakeModel("递归就像照镜子……")
    tutor = TutorAgent(model)

    assert tutor.explain("递归") == "递归就像照镜子……"


def test_tutor_puts_topic_in_prompt() -> None:
    model = FakeModel("讲解")
    TutorAgent(model).explain("递归的终止条件")

    assert "递归的终止条件" in model.prompts[0]


def test_tutor_without_context_says_beginner() -> None:
    """没有历史记录时要明说「按零基础处理」。

    不说的话，模型可能自己编一个学生画像出来，那比不个性化更糟。
    """
    model = FakeModel("讲解")
    TutorAgent(model).explain("递归")

    assert "零基础" in model.prompts[0]


def test_tutor_context_is_injected() -> None:
    """学生情况要真的进提示词——这是导师 Agent 存在的理由。"""
    model = FakeModel("讲解")
    TutorAgent(model).explain("递归", context="- 待加强：递归（掌握度 0.20，练过 5 题对 2 题）")

    prompt = model.prompts[0]
    assert "掌握度 0.20" in prompt
    assert "零基础" not in prompt


def test_tutor_blank_topic_rejected() -> None:
    with pytest.raises(ValueError, match="知识点"):
        TutorAgent(FakeModel("讲解")).explain("   ")


def test_tutor_strips_whitespace_from_reply() -> None:
    """模型爱在回答前后加空行，去掉再返回。"""
    tutor = TutorAgent(FakeModel("\n\n  讲解正文  \n\n"))

    assert tutor.explain("递归") == "讲解正文"


# --------------------------------------------------------------------------
# 出题 Agent
# --------------------------------------------------------------------------

SAMPLE_QUESTIONS = [
    {"question": "写一个阶乘函数", "answer": "def f(n): return 1 if n <= 1 else n * f(n-1)", "hint": "注意终止条件"},
    {"question": "递归必须有哪两部分", "answer": "基线条件与递归条件", "hint": "少了会怎样"},
]


def test_quiz_generates_questions() -> None:
    model = FakeModel(json_text(SAMPLE_QUESTIONS))
    questions = QuizAgent(model).generate("递归")

    assert len(questions) == 2
    assert questions[0]["question"] == "写一个阶乘函数"
    assert questions[0]["hint"] == "注意终止条件"


def test_quiz_parses_fenced_json() -> None:
    """模型爱用 Markdown 围栏包 JSON。"""
    model = FakeModel(f"```json\n{json_text(SAMPLE_QUESTIONS)}\n```")

    assert len(QuizAgent(model).generate("递归")) == 2


def test_quiz_parses_json_with_explanation() -> None:
    model = FakeModel(f"好的，这是题目：{json_text(SAMPLE_QUESTIONS)} 加油！")

    assert len(QuizAgent(model).generate("递归")) == 2


def test_quiz_respects_count_limit() -> None:
    """就算模型多出了几道，也只取要求的数量。"""
    many = [{"question": f"第{i}题", "answer": "答案"} for i in range(10)]
    questions = QuizAgent(FakeModel(json_text(many))).generate("递归", count=3)

    assert len(questions) == 3


def test_quiz_drops_question_without_answer() -> None:
    """没有参考答案的题目批改不了，必须丢掉。

    宁可少一道题，也不能出一道没法批的题——那会让评估环节直接卡住。
    """
    payload = [
        {"question": "没答案的题"},
        {"question": "有答案的题", "answer": "答案"},
    ]
    questions = QuizAgent(FakeModel(json_text(payload))).generate("递归")

    assert [item["question"] for item in questions] == ["有答案的题"]


def test_quiz_drops_question_without_text() -> None:
    payload = [{"answer": "光有答案没有题干"}, {"question": "正常的题", "answer": "答案"}]
    questions = QuizAgent(FakeModel(json_text(payload))).generate("递归")

    assert len(questions) == 1


def test_quiz_missing_hint_defaults_to_empty() -> None:
    """hint 是可选信息，不给也不该报错。"""
    questions = QuizAgent(FakeModel(json_text([{"question": "题", "answer": "答"}]))).generate("递归")

    assert questions[0]["hint"] == ""


def test_quiz_unparsable_raises() -> None:
    """一道题都解析不出来要报错。

    返回空列表的话，上层以为「出好了，做吧」，然后拿着空题单去找学生。
    """
    with pytest.raises(ValueError, match="解析"):
        QuizAgent(FakeModel("抱歉，我出不了题。")).generate("递归")


def test_quiz_blank_topic_rejected() -> None:
    with pytest.raises(ValueError, match="知识点"):
        QuizAgent(FakeModel(json_text(SAMPLE_QUESTIONS))).generate("  ")


def test_quiz_rejects_non_positive_count() -> None:
    """要 0 道题或 -1 道题，是调用方写错了，得说清楚。"""
    with pytest.raises(ValueError, match="数量"):
        QuizAgent(FakeModel(json_text(SAMPLE_QUESTIONS))).generate("递归", count=0)


def test_quiz_context_makes_prompt_targeted() -> None:
    """带了学生情况，题目要针对薄弱点出。"""
    model = FakeModel(json_text(SAMPLE_QUESTIONS))
    QuizAgent(model).generate("递归", context="- 待加强：递归的终止条件")

    assert "递归的终止条件" in model.prompts[0]


# --------------------------------------------------------------------------
# 评估 Agent
# --------------------------------------------------------------------------


def test_grade_correct_answer() -> None:
    model = FakeModel(json_text({"correct": True, "score": 1.0, "feedback": "基线条件说对了"}))
    result = GraderAgent(model).grade("递归要有哪两部分？", reference="基线条件和递归条件", answer="基线条件和递归条件")

    assert result["correct"] is True
    assert result["score"] == 1.0
    assert result["feedback"] == "基线条件说对了"


def test_grade_wrong_answer() -> None:
    model = FakeModel(json_text({"correct": False, "score": 0.0, "feedback": "漏了基线条件"}))
    result = GraderAgent(model).grade("递归要有哪两部分？", reference="基线条件和递归条件", answer="递归条件")

    assert result["correct"] is False
    assert result["score"] == 0.0


def test_grade_accepts_chinese_boolean() -> None:
    """模型把 true 写成「正确」也得认。"""
    model = FakeModel(json_text({"correct": "正确", "feedback": "没问题"}))
    result = GraderAgent(model).grade("题", reference="参", answer="答")

    assert result["correct"] is True


def test_grade_accepts_numeric_boolean() -> None:
    """写成 1 / 0 也要认。"""
    assert GraderAgent(FakeModel(json_text({"correct": 1}))).grade("题", reference="参", answer="答")["correct"] is True
    assert GraderAgent(FakeModel(json_text({"correct": 0}))).grade("题", reference="参", answer="答")["correct"] is False


def test_grade_accepts_string_score() -> None:
    """分数写成字符串是常见情况。"""
    model = FakeModel(json_text({"correct": True, "score": "0.5"}))
    assert GraderAgent(model).grade("题", reference="参", answer="答")["score"] == 0.5


def test_grade_score_falls_back_to_correct() -> None:
    """没给分数就按对错推：对是 1 分，错是 0 分。"""
    model = FakeModel(json_text({"correct": True}))
    assert GraderAgent(model).grade("题", reference="参", answer="答")["score"] == 1.0


def test_grade_clamps_out_of_range_score() -> None:
    """分数越界就夹回区间，不能让它流进画像。

    这是模型给的分，不是程序参数——夹紧比中断整个学习流程合理。
    """
    model = FakeModel(json_text({"correct": True, "score": 1.5}))
    assert GraderAgent(model).grade("题", reference="参", answer="答")["score"] == 1.0

    model = FakeModel(json_text({"correct": False, "score": -0.5}))
    assert GraderAgent(model).grade("题", reference="参", answer="答")["score"] == 0.0


def test_grade_unusable_score_falls_back() -> None:
    """分数是 NaN 或一句废话时，按对错推算。"""
    for bad in ["nan", "不知道", None]:
        model = FakeModel(json_text({"correct": True, "score": bad}))
        assert GraderAgent(model).grade("题", reference="参", answer="答")["score"] == 1.0


def test_grade_falls_back_to_text_when_json_broken() -> None:
    """JSON 坏了，退一步从文字里认对错。"""
    model = FakeModel("这道题回答正确，思路清晰。")
    result = GraderAgent(model).grade("题", reference="参", answer="答")

    assert result["correct"] is True
    assert "正确" in result["feedback"]


def test_grade_negation_detected_before_positive() -> None:
    """「不正确」必须判成错。

    肯定词表里有「正确」，否定词表里有「不正确」——先查否定才不会读反。
    这个顺序错了会出静默误判，光看代码很难发现。
    """
    model = FakeModel("这个回答不正确。")
    assert GraderAgent(model).grade("题", reference="参", answer="答")["correct"] is False


def test_grade_wrong_keyword_detected() -> None:
    model = FakeModel("答案错误，漏了终止条件。")
    assert GraderAgent(model).grade("题", reference="参", answer="答")["correct"] is False


def test_grade_positive_before_negative_wins() -> None:
    """「正确，但表述可以更简洁」里正确在前，算对。

    带转折的评语很常见，不能因为出现「但」就判错。
    """
    model = FakeModel("回答正确，但表述可以更简洁一点。")
    assert GraderAgent(model).grade("题", reference="参", answer="答")["correct"] is True


def test_grade_unjudgeable_raises() -> None:
    """彻底判不出来时报错，绝不默默判错。

    默默判错会污染画像，而且事后完全查不出是哪一次判错的。
    """
    model = FakeModel("嗯……这道题的情况比较复杂，我们下次再讨论吧。")

    with pytest.raises(ValueError, match="无法判断"):
        GraderAgent(model).grade("题", reference="参", answer="答")


def test_grade_blank_question_rejected() -> None:
    with pytest.raises(ValueError, match="题干"):
        GraderAgent(FakeModel("x")).grade("   ", reference="参", answer="答")


def test_grade_blank_answer_rejected() -> None:
    """空答案要报错。

    直接判「答错」看似合理，但空答案也可能是程序传错了参数——
    静默记一次答错会污染画像，不如报错让人看一眼。
    """
    with pytest.raises(ValueError, match="答案"):
        GraderAgent(FakeModel("x")).grade("题", reference="参", answer="   ")


def test_grade_without_reference_still_works() -> None:
    """没有参考答案时也要能批（提示词里会说明让模型自行判断）。"""
    model = FakeModel(json_text({"correct": True, "feedback": "可以"}))
    result = GraderAgent(model).grade("题", reference="", answer="答")

    assert result["correct"] is True
    assert "自行判断" in model.prompts[0] or "未提供" in model.prompts[0]


def test_grade_keeps_raw_text_when_feedback_missing() -> None:
    """模型只判对错、不写评语时，把原文留下来当反馈，别丢信息。"""
    model = FakeModel("判断：正确")
    result = GraderAgent(model).grade("题", reference="参", answer="答")

    assert result["feedback"] == "判断：正确"
