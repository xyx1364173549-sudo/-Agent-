"""实验指标测试。

指标是论文里最不该出错的一类代码——算错了不会报错，只会悄悄给出
一个偏高的分数，整章结论跟着站不住。所以这里逐条盯边界：
除零、空输入、名次与倒数的对应关系、"练习前"还是"练习后"的取值。
"""

from __future__ import annotations

import pytest

from src.eval.metrics import (
    coverage,
    efficiency,
    hit_at_k,
    idle_ratio,
    match_keywords,
    mean,
    recall_at_k,
    reciprocal_rank,
    std,
    summarize,
)

# --------------------------------------------------------------------------
# 检索指标
# --------------------------------------------------------------------------


def test_hit_at_k_found() -> None:
    assert hit_at_k(["a.md", "b.md"], {"b.md"}, k=3) == 1.0


def test_hit_at_k_not_found() -> None:
    assert hit_at_k(["a.md", "b.md"], {"c.md"}, k=3) == 0.0


def test_hit_at_k_respects_k() -> None:
    """相关文档排在第 3 名，但只看前 2 条时算未命中。"""
    assert hit_at_k(["a.md", "b.md", "c.md"], {"c.md"}, k=2) == 0.0
    assert hit_at_k(["a.md", "b.md", "c.md"], {"c.md"}, k=3) == 1.0


def test_hit_at_k_rejects_bad_k() -> None:
    with pytest.raises(ValueError, match="k 需为正整数"):
        hit_at_k(["a.md"], {"a.md"}, k=0)


def test_hit_at_k_requires_relevant() -> None:
    """没有标注就算不出指标，报错比返回 0 诚实。"""
    with pytest.raises(ValueError, match="相关标注"):
        hit_at_k(["a.md"], set(), k=1)


def test_reciprocal_rank_first() -> None:
    assert reciprocal_rank(["a.md", "b.md"], {"a.md"}) == 1.0


def test_reciprocal_rank_third() -> None:
    assert reciprocal_rank(["a.md", "b.md", "c.md"], {"c.md"}) == pytest.approx(1 / 3, abs=1e-6)


def test_reciprocal_rank_miss() -> None:
    assert reciprocal_rank(["a.md"], {"z.md"}) == 0.0


def test_reciprocal_rank_uses_best_rank() -> None:
    """同一相关文档出现多次时，取最靠前的那个名次。"""
    assert reciprocal_rank(["a.md", "b.md", "b.md"], {"b.md"}) == 0.5


def test_recall_at_k_partial() -> None:
    """相关文档有两篇，只找回来一篇 —— 召回率 0.5。

    这正是 Hit@k 看不出来的地方：它只管"有没有命中"，不看"命中了几成"。
    """
    assert recall_at_k(["a.md", "c.md"], {"a.md", "b.md"}, k=2) == 0.5


def test_recall_at_k_full() -> None:
    assert recall_at_k(["a.md", "b.md"], {"a.md", "b.md"}, k=2) == 1.0


def test_recall_at_k_no_duplicates_counted_twice() -> None:
    """同一篇文档返回两次只能算一篇。

    不去重的话，召回的分子会被同一篇重复撑大，出现 1.5 这种分数。
    """
    assert recall_at_k(["a.md", "a.md"], {"a.md", "b.md"}, k=2) == 0.5


# --------------------------------------------------------------------------
# 要点覆盖
# --------------------------------------------------------------------------


def test_coverage_all_hit() -> None:
    assert coverage([True, True, True]) == 1.0


def test_coverage_partial() -> None:
    assert coverage([True, False, True, False]) == 0.5


def test_coverage_empty() -> None:
    """没有要点时返回 0 而不是崩掉，也不该返回 1（那是白送分）。"""
    assert coverage([]) == 0.0


def test_match_keywords_hits() -> None:
    points = ["终止条件是递归的出口"]
    answer = "递归必须有终止条件，否则会一直调用下去。"

    assert match_keywords(answer, points) == [True]


def test_match_keywords_misses() -> None:
    assert match_keywords("今天天气不错", ["终止条件是递归的出口"]) == [False]


def test_match_keywords_ignores_filler_words() -> None:
    """要点里的虚词不能被拿来匹配。

    「提到终止条件」里的「提到」在回答中永远不会原样出现，
    不剔掉的话这一条永远判不命中，分数恒为 0。
    """
    points = ["提到终止条件"]
    answer = "递归要有终止条件。"

    assert match_keywords(answer, points) == [True]


def test_match_keywords_multiple_points() -> None:
    points = ["终止条件是递归的出口", "调用栈会一直堆下去"]
    answer = "递归必须有终止条件；没有出口的话调用栈会一直堆下去直到崩溃。"

    assert match_keywords(answer, points) == [True, True]


# --------------------------------------------------------------------------
# 学习过程指标
# --------------------------------------------------------------------------


def test_efficiency_improvement() -> None:
    """比基线少花三成轮数。"""
    assert efficiency(rounds=7, baseline_rounds=10) == pytest.approx(0.3, abs=1e-6)


def test_efficiency_no_improvement() -> None:
    assert efficiency(rounds=10, baseline_rounds=10) == 0.0


def test_efficiency_worse_than_baseline() -> None:
    """比基线还差时是负数，如实反映，不要夹到 0。"""
    assert efficiency(rounds=12, baseline_rounds=10) == pytest.approx(-0.2, abs=1e-6)


def test_efficiency_rejects_zero_baseline() -> None:
    with pytest.raises(ValueError, match="基线"):
        efficiency(rounds=1, baseline_rounds=0)


def test_idle_ratio_uses_mastery_before() -> None:
    """判「本来就会」要用**练习前**的掌握度。

    这条盯的是一个很容易搞错的地方：练完之后掌握度当然上去了，
    拿练之后的值判断，无效练习永远是 0，指标就废了。
    """
    attempts = [
        {"mastery_before": 0.9},  # 本来就会 -> 白练
        {"mastery_before": 0.2},  # 确实需要练
        {"mastery_before": 0.8},  # 本来就会 -> 白练
        {"mastery_before": 0.1},  # 确实需要练
    ]

    assert idle_ratio(attempts) == 0.5


def test_idle_ratio_empty() -> None:
    assert idle_ratio([]) == 0.0


def test_idle_ratio_custom_threshold() -> None:
    attempts = [{"mastery_before": 0.5, "threshold": 0.4}]
    assert idle_ratio(attempts) == 1.0


# --------------------------------------------------------------------------
# 统计工具
# --------------------------------------------------------------------------


def test_mean() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_mean_empty() -> None:
    """空列表返回 0，不让它除零崩掉。"""
    assert mean([]) == 0.0


def test_std_needs_two_values() -> None:
    """单个值算不出波动，返回 0 而不是崩。"""
    assert std([0.5]) == 0.0
    assert std([]) == 0.0


def test_std_known_value() -> None:
    assert std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) == pytest.approx(2.1381, abs=1e-3)


def test_summarize() -> None:
    result = summarize([1.0, 2.0, 3.0])

    assert result["mean"] == 2.0
    assert result["min"] == 1.0
    assert result["max"] == 3.0
    assert result["n"] == 3
    assert result["std"] == 1.0


def test_summarize_empty() -> None:
    result = summarize([])

    assert result["mean"] == 0.0
    assert result["n"] == 0
