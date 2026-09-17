"""大模型输出解析测试。

这个模块的价值全在「模型不按格式说话」的时候，所以测试用例照着真实毛病写：
围栏、客套话、多个 JSON 片段、缺引号、包一层壳。
"""

from __future__ import annotations

import pytest

from src.utils.json_parse import (
    extract_json,
    extract_json_array,
    extract_json_object,
    strip_fence,
)


# --------------------------------------------------------------------------
# 去围栏
# --------------------------------------------------------------------------


def test_strip_fence_json() -> None:
    assert strip_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_fence_plain() -> None:
    assert strip_fence("```\n[1, 2]\n```") == "[1, 2]"


def test_strip_fence_without_fence() -> None:
    assert strip_fence('  {"a": 1}  ') == '{"a": 1}'


def test_strip_fence_keeps_surrounding_text() -> None:
    """没有围栏时不能乱切，原文（含客套话）留着给后面的解析器处理。"""
    assert strip_fence("这是结果：{\"a\": 1} 完毕") == "这是结果：{\"a\": 1} 完毕"


# --------------------------------------------------------------------------
# 提取 JSON
# --------------------------------------------------------------------------


def test_extract_pure_json_object() -> None:
    assert extract_json('{"correct": true}') == {"correct": True}


def test_extract_pure_json_array() -> None:
    assert extract_json('[{"topic": "递归"}]') == [{"topic": "递归"}]


def test_extract_json_with_leading_text() -> None:
    """模型爱在 JSON 前面说一句。"""
    assert extract_json('好的，结果是：{"correct": true}') == {"correct": True}


def test_extract_json_with_trailing_text() -> None:
    assert extract_json('{"correct": true} 希望对你有帮助。') == {"correct": True}


def test_extract_json_with_both_sides() -> None:
    assert extract_json('好的：{"a": 1} 完毕') == {"a": 1}


def test_extract_json_ignores_earlier_broken_fragment() -> None:
    """模型先给个坏掉的示例、再给正式答案时要能挑出能解析的那个。

    这正是「取第一个 [ 到最后一个 ]」那种做法会栽的地方：
    它会把两段黏在一起，结果必然解析失败。
    """
    text = '示例（不用管）：[1, 2 这不是合法 JSON。正式结果：{"correct": false}'

    assert extract_json(text) == {"correct": False}


def test_extract_json_prefers_whole_document() -> None:
    """整个文档就是 JSON 时不该被后面的干扰。"""
    assert extract_json('{"items": [1, 2, 3]}') == {"items": [1, 2, 3]}


def test_extract_json_empty_string() -> None:
    assert extract_json("") is None


def test_extract_json_no_json_at_all() -> None:
    assert extract_json("抱歉，我做不到。") is None


def test_extract_json_incomplete_json() -> None:
    """截断的 JSON 不该被硬凑出来，返回 None 更诚实。"""
    assert extract_json('{"correct": tru') is None


# --------------------------------------------------------------------------
# 只要数组 / 只要对象
# --------------------------------------------------------------------------


def test_extract_array_basic() -> None:
    result = extract_json_array('[{"topic": "递归"}, {"topic": "二分"}]')
    assert len(result) == 2


def test_extract_array_filters_non_dict() -> None:
    """数组里混进字符串、数字等杂物，过滤掉。"""
    assert extract_json_array('["递归", 42, {"topic": "二分"}]') == [{"topic": "二分"}]


def test_extract_array_unwraps_object() -> None:
    """模型把数组包在对象里，例如 {"topics": [...]}。"""
    assert extract_json_array('{"topics": [{"topic": "递归"}]}') == [{"topic": "递归"}]


def test_extract_array_from_object_without_list() -> None:
    assert extract_json_array('{"answer": "42"}') == []


def test_extract_array_from_object_given_none() -> None:
    assert extract_json_array("没有 JSON") == []


def test_extract_object_basic() -> None:
    assert extract_json_object('{"correct": true, "score": 1}') == {"correct": True, "score": 1}


def test_extract_object_from_array() -> None:
    """模型把对象包在数组里，例如 [{...}]。"""
    assert extract_json_object('[{"correct": true}]') == {"correct": True}


def test_extract_object_from_array_skips_noise() -> None:
    assert extract_json_object('["说明", {"correct": true}]') == {"correct": True}


def test_extract_object_given_array_of_scalars() -> None:
    assert extract_json_object("[1, 2, 3]") == {}


def test_extract_object_given_none() -> None:
    assert extract_json_object("没有 JSON") == {}


# --------------------------------------------------------------------------
# 边界
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_blank_inputs(text: str) -> None:
    assert extract_json(text) is None
    assert extract_json_array(text) == []
    assert extract_json_object(text) == {}
