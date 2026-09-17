"""进度看板解析与渲染测试。

``PLAN.md`` 是项目进度的唯一权威来源，因此它同时也是**一份契约**：
结构一旦被改坏，看板就会静默失真。这些用例把契约钉死。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.gen_progress import (
    Module,
    Plan,
    Task,
    format_summary,
    generate,
    parse_plan,
    render_html,
)

MINIMAL_PLAN = """# 测试计划

## [M0] 模块零
<!-- section: 全局前置 -->
- [ ] M0.1 待办任务 — 交付：产物 A
- [~] M0.2 进行中任务 — 交付：产物 B
- [x] M0.3 已完成任务 — 交付：产物 C
- [!] M0.4 阻塞任务

## [M1] 模块一
<!-- section: 第 1 章 -->
- [x] M1.1 完成
"""


@pytest.fixture
def sample_plan(tmp_path: Path) -> Path:
    path = tmp_path / "PLAN.md"
    path.write_text(MINIMAL_PLAN, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# 解析
# --------------------------------------------------------------------------


def test_parse_modules_and_tasks(sample_plan: Path) -> None:
    plan = parse_plan(sample_plan)
    assert [m.code for m in plan.modules] == ["M0", "M1"]
    assert plan.modules[0].name == "模块零"
    assert plan.modules[0].section == "全局前置"
    assert len(plan.modules[0].tasks) == 4


@pytest.mark.parametrize(
    ("index", "status"),
    [(0, "todo"), (1, "doing"), (2, "done"), (3, "blocked")],
)
def test_parse_status_flags(sample_plan: Path, index: int, status: str) -> None:
    assert parse_plan(sample_plan).modules[0].tasks[index].status == status


def test_deliverable_is_split_out(sample_plan: Path) -> None:
    task = parse_plan(sample_plan).modules[0].tasks[0]
    assert task.title == "待办任务"
    assert task.deliverable == "产物 A"


def test_task_without_deliverable_keeps_full_title(sample_plan: Path) -> None:
    task = parse_plan(sample_plan).modules[0].tasks[3]
    assert task.deliverable == ""
    assert task.title == "阻塞任务"


def test_plain_text_lines_are_ignored(sample_plan: Path, tmp_path: Path) -> None:
    """说明文字、表格、变更记录都不该被误读成任务。"""
    path = tmp_path / "PLAN2.md"
    path.write_text(
        MINIMAL_PLAN + "\n## 变更记录\n\n| 日期 | 变更 |\n| --- | --- |\n| 今天 | 建表 |\n",
        encoding="utf-8",
    )
    plan = parse_plan(path)
    assert len(plan.modules) == 2
    assert plan.total == 5


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_plan(tmp_path / "nope.md")


def test_empty_plan_does_not_crash(tmp_path: Path) -> None:
    path = tmp_path / "EMPTY.md"
    path.write_text("# 空计划\n", encoding="utf-8")
    plan = parse_plan(path)
    assert plan.modules == []
    assert plan.progress == 0.0
    assert plan.current_module is None


def test_module_section_only_read_right_after_title(tmp_path: Path) -> None:
    """正文里出现的同形注释不该被当成章节标注。"""
    path = tmp_path / "PLAN3.md"
    path.write_text(
        "# t\n\n## [M0] 模块零\n\n普通说明\n<!-- section: 不该生效 -->\n- [ ] M0.1 任务\n",
        encoding="utf-8",
    )
    assert parse_plan(path).modules[0].section == ""


def test_first_task_survives_without_section_comment(tmp_path: Path) -> None:
    """回归用例：模块标题下面若没写 section 注释，首条任务不能被解析器吞掉。

    这个 bug 曾经真实存在——解析 section 的分支无条件 ``continue``，
    把标题后的第一行直接吃掉了。表现是「任务总数莫名少一条」。
    """
    path = tmp_path / "REGRESSION.md"
    path.write_text(
        "# t\n\n## [M0] 模块\n- [ ] M0.1 第一条任务\n- [ ] M0.2 第二条任务\n",
        encoding="utf-8",
    )
    module = parse_plan(path).modules[0]
    assert [t.code for t in module.tasks] == ["M0.1", "M0.2"]
    assert module.total == 2


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------


def test_progress_counts_doing_as_half(sample_plan: Path) -> None:
    """权重算法：done=1、doing=0.5、todo 与 blocked=0。

    sample_plan 共 5 条：M0 的 todo+doing+done+blocked 合计 1.5，
    M1 的 1 条 done 合计 1.0，总计 2.5 / 5 = 50%。
    注意这与「完成条数占比」不是一回事——完成条数占比是 2/5 = 40%。
    """
    plan = parse_plan(sample_plan)
    assert plan.total == 5
    assert plan.done == 2
    assert plan.doing == 1
    assert plan.todo == 1
    assert plan.blocked == 1
    assert plan.progress == pytest.approx(50.0)


def test_module_progress(sample_plan: Path) -> None:
    module = parse_plan(sample_plan).modules[0]
    assert module.total == 4
    assert module.progress == pytest.approx(37.5)
    assert module.is_finished is False


def test_module_is_finished(sample_plan: Path) -> None:
    assert parse_plan(sample_plan).modules[1].is_finished is True


def test_current_module_is_first_unfinished(sample_plan: Path) -> None:
    assert parse_plan(sample_plan).current_module.code == "M0"  # type: ignore[union-attr]


def test_current_module_none_when_all_done(tmp_path: Path) -> None:
    path = tmp_path / "DONE.md"
    path.write_text("# t\n\n## [M0] 全部完成\n- [x] M0.1 任务\n", encoding="utf-8")
    assert parse_plan(path).current_module is None


def test_empty_module_progress_is_zero() -> None:
    assert Module(code="M9", name="空", section="").progress == 0.0


def test_task_status_label_and_icon() -> None:
    task = Task(code="M0.1", title="t", status="done")
    assert task.status_label == "已完成"
    assert task.icon == "✓"


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------


def test_render_html_contains_key_sections(sample_plan: Path) -> None:
    plan = parse_plan(sample_plan)
    html = render_html(plan)
    assert "<!DOCTYPE html>" in html
    assert "M0" in html and "模块零" in html
    assert "50.0" in html


def test_render_html_is_deterministic(sample_plan: Path) -> None:
    """同一个 PLAN.md 渲染两次，结果必须逐字节相同。

    看板页脚曾经写着「生成时间」，于是每次重新生成都有 diff，
    `check.py` 一跑就多出一堆只改时间戳的提交。这里守住幂等性。
    """
    plan = parse_plan(sample_plan)
    assert render_html(plan) == render_html(plan)


def test_render_html_has_no_timestamp(sample_plan: Path) -> None:
    """页脚不该出现「生成时间」——它是渲染不幂等的唯一来源。"""
    html = render_html(parse_plan(sample_plan))
    assert "生成时间" not in html


def test_render_html_escapes_untrusted_text(tmp_path: Path) -> None:
    """任务描述里的尖括号必须转义，否则看板会被写坏。"""
    path = tmp_path / "XSS.md"
    path.write_text("# t\n\n## [M0] 模块\n- [ ] M0.1 <script>alert(1)</script>\n", encoding="utf-8")
    html = render_html(parse_plan(path))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_html_handles_empty_plan(tmp_path: Path) -> None:
    path = tmp_path / "E.md"
    path.write_text("# 空\n", encoding="utf-8")
    html = render_html(Plan(title="空", modules=[]))
    assert "还没有模块" in html


def test_generate_writes_file(sample_plan: Path, tmp_path: Path) -> None:
    output = tmp_path / "docs" / "progress.html"
    path, plan = generate(sample_plan, output)
    assert path == output
    assert output.exists()
    assert plan.total == 5


def test_format_summary_lists_every_module(sample_plan: Path) -> None:
    summary = format_summary(parse_plan(sample_plan))
    assert "M0" in summary and "M1" in summary
    assert "50.0%" in summary


# --------------------------------------------------------------------------
# 真实计划表契约
# --------------------------------------------------------------------------


def test_real_plan_is_well_formed(project_root: Path) -> None:
    """防止 PLAN.md 被改坏：模块齐全、每个模块都有子任务、统计自洽。"""
    plan = parse_plan(project_root / "PLAN.md")
    assert len(plan.modules) == 9, "模块数量应为 M0–M8"
    assert plan.total >= 40
    assert all(m.tasks for m in plan.modules), "不允许出现空模块"
    assert plan.done + plan.doing + plan.todo + plan.blocked == plan.total
    assert 0.0 <= plan.progress <= 100.0


def test_real_plan_task_codes_are_consistent(project_root: Path) -> None:
    """任务编号前缀必须与所属模块一致，否则看板会出现「M2 里躺着 M3.1」。"""
    plan = parse_plan(project_root / "PLAN.md")
    for module in plan.modules:
        for task in module.tasks:
            assert task.code.startswith(module.code + "."), f"{module.code} 下出现了 {task.code}"
