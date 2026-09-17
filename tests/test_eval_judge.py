"""要点判定器测试。

判定器要用真实模型跑，所以这里全部用假模型——重点是**模型输出不听话时**
怎么办：数量对不上、写成中文、裹着说明文字、干脆给一坨垃圾。

评测脚本跑到一半崩掉、还得从头再花钱跑一遍，代价比少判一条大得多。
所以这里的策略是「尽量认，认不出就补 False 并留警告」，而不是抛错。
"""

from __future__ import annotations

import json

import pytest

from src.eval.judge import judge_points
from src.utils.json_parse import extract_json


class FakeReply:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeReply:
        self.prompts.append(prompt)
        return FakeReply(self.content)


POINTS = ["指出薄弱点在终止条件", "提到练习的正确率", "给出具体的学习动作"]


def test_judge_basic() -> None:
    model = FakeModel("[true, false, true]")

    assert judge_points("我该学什么", "回答内容", POINTS, model=model) == [True, False, True]


def test_judge_returns_length_matching_points() -> None:
    marks = judge_points("问", "答", POINTS, model=FakeModel("[true]"))

    assert len(marks) == len(POINTS)


def test_judge_accepts_fenced_json() -> None:
    """模型爱用 Markdown 围栏包 JSON。"""
    model = FakeModel("```json\n[true, true, false]\n```")

    assert judge_points("问", "答", POINTS, model=model) == [True, True, False]


def test_judge_accepts_surrounding_text() -> None:
    model = FakeModel("判断结果如下：[true, false, false] 希望有帮助。")

    assert judge_points("问", "答", POINTS, model=model) == [True, False, False]


def test_judge_accepts_chinese_booleans() -> None:
    """模型写成「是 / 否」也要认。"""
    model = FakeModel('["是", "否", "是"]')

    assert judge_points("问", "答", POINTS, model=model) == [True, False, True]


def test_judge_accepts_numeric_booleans() -> None:
    model = FakeModel("[1, 0, 1]")

    assert judge_points("问", "答", POINTS, model=model) == [True, False, True]


def test_judge_pads_missing_marks_with_false() -> None:
    """模型少给几条时补 False，不抛错。

    补 False 的偏向是保守的：宁可低估覆盖度，也不要把没提到的要点算成提到了。
    报错的话评测脚本会中途崩掉，整轮白跑。
    """
    model = FakeModel("[true]")

    assert judge_points("问", "答", POINTS, model=model) == [True, False, False]


def test_judge_truncates_extra_marks() -> None:
    """模型多给了几条时截断。"""
    model = FakeModel("[true, true, true, true, true]")

    assert judge_points("问", "答", POINTS, model=model) == [True, True, True]


def test_judge_unparsable_returns_all_false() -> None:
    """完全解析不出来时全判未覆盖，并如实返回等长列表。"""
    model = FakeModel("这道题很难判断，我建议你换个问法。")

    assert judge_points("问", "答", POINTS, model=model) == [False, False, False]


def test_judge_drops_unrecognizable_items() -> None:
    """数组里混进认不出的值时跳过它，剩下的照常判读。"""
    model = FakeModel('["不知道", true, false, true]')

    marks = judge_points("问", "答", POINTS, model=model)

    assert marks[:3] == [True, False, True]


def test_judge_empty_points_short_circuits() -> None:
    """没有要点就直接返回，不该白调一次模型。"""
    model = FakeModel("[true]")

    assert judge_points("问", "答", [], model=model) == []
    assert model.prompts == []


def test_judge_prompt_contains_points_and_count() -> None:
    """要点和数量都要进提示词，模型才知道该判几条。"""
    model = FakeModel("[true, true, true]")
    judge_points("我的问题", "回答内容", POINTS, model=model)

    prompt = model.prompts[0]
    assert "我的问题" in prompt
    assert "回答内容" in prompt
    assert POINTS[0] in prompt
    assert "正好是 3 个" in prompt


def test_judge_handles_blank_question() -> None:
    """没有问题时也要能判，不该崩。"""
    model = FakeModel("[true, true, true]")

    assert judge_points("", "答", POINTS, model=model) == [True, True, True]


def test_extract_json_helper_still_used() -> None:
    """确认判定器复用的是公共解析器，而不是自己又写一份。

    两处各写一份解析逻辑的话，将来修好一处、另一处还留着老毛病。
    """
    assert extract_json("[true]") == [True]
