"""进度看板生成器。

**数据源**：``PLAN.md``（唯一权威，人工维护）
**产物**：``docs/progress.html``（自包含深色主题看板，自动生成）

这样设计的理由：进度只有一处需要维护——计划表本身。看板是派生视图，
永远不会和计划表对不上。手工维护两份进度必然会不一致。

``PLAN.md`` 中需要遵守的书写约定：

- 模块标题：``## [M0] 工程地基 · 项目骨架``
- 章节标注：紧随标题的一行 ``<!-- section: 全局前置 -->``
- 子任务：``- [ ] M0.1 描述 — 交付：产物``，方括号内 `` ``/``x``/``~``/``!``
  分别表示 待办 / 已完成 / 进行中 / 阻塞

用法::

    python scripts/gen_progress.py            # 生成 docs/progress.html
    python scripts/gen_progress.py --print    # 同时在终端打印进度摘要
"""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLAN = PROJECT_ROOT / "PLAN.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "progress.html"

# 复选框符号 -> 内部状态
STATUS_MAP: dict[str, str] = {" ": "todo", "x": "done", "~": "doing", "!": "blocked"}
STATUS_LABEL: dict[str, str] = {"todo": "待办", "done": "已完成", "doing": "进行中", "blocked": "阻塞"}
STATUS_ICON: dict[str, str] = {"todo": "○", "done": "✓", "doing": "◐", "blocked": "✕"}
# 进度计算权重：进行中的任务按半条计
STATUS_WEIGHT: dict[str, float] = {"done": 1.0, "doing": 0.5, "todo": 0.0, "blocked": 0.0}

MODULE_RE = re.compile(r"^##\s+\[([A-Z]\d+)\]\s+(.+?)\s*$")
SECTION_RE = re.compile(r"^<!--\s*section:\s*(.+?)\s*-->$")
TASK_RE = re.compile(r"^-\s+\[([ x~!])\]\s+([A-Z]\d+\.\d+)\s+(.+?)\s*$")
DELIVER_RE = re.compile(r"\s+[—–]{1,2}\s*交付[:：]\s*(.+)$")


@dataclass
class Task:
    """一条子任务。"""

    code: str
    title: str
    status: str
    deliverable: str = ""

    @property
    def status_label(self) -> str:
        return STATUS_LABEL[self.status]

    @property
    def icon(self) -> str:
        return STATUS_ICON[self.status]


@dataclass
class Module:
    """一个模块（对应论文的一个章节或一段前置工作）。"""

    code: str
    name: str
    section: str
    tasks: list[Task] = field(default_factory=list)

    # ---- 统计 ----

    @property
    def total(self) -> int:
        return len(self.tasks)

    @property
    def done(self) -> int:
        return sum(1 for t in self.tasks if t.status == "done")

    def count(self, status: str) -> int:
        return sum(1 for t in self.tasks if t.status == status)

    @property
    def progress(self) -> float:
        """按权重计算的完成度，取值 0.0 ~ 100.0。"""
        if not self.tasks:
            return 0.0
        return round(sum(STATUS_WEIGHT[t.status] for t in self.tasks) / self.total * 100, 1)

    @property
    def is_finished(self) -> bool:
        return bool(self.tasks) and self.done == self.total


@dataclass
class Plan:
    """整份计划表。"""

    title: str
    modules: list[Module] = field(default_factory=list)
    source: Path | None = None

    @property
    def tasks(self) -> list[Task]:
        return [t for module in self.modules for t in module.tasks]

    @property
    def total(self) -> int:
        return len(self.tasks)

    def count(self, status: str) -> int:
        return sum(1 for t in self.tasks if t.status == status)

    @property
    def done(self) -> int:
        return self.count("done")

    @property
    def doing(self) -> int:
        return self.count("doing")

    @property
    def todo(self) -> int:
        return self.count("todo")

    @property
    def blocked(self) -> int:
        return self.count("blocked")

    @property
    def progress(self) -> float:
        if not self.tasks:
            return 0.0
        return round(sum(STATUS_WEIGHT[t.status] for t in self.tasks) / self.total * 100, 1)

    @property
    def current_module(self) -> Module | None:
        """第一个尚未完成的模块——也就是「当前正在做的模块」。"""
        for module in self.modules:
            if not module.is_finished:
                return module
        return None


# --------------------------------------------------------------------------
# 解析
# --------------------------------------------------------------------------


def parse_plan(path: Path) -> Plan:
    """把 PLAN.md 解析成结构化对象。

    容错策略：只识别符合约定的行，其余内容（总览表格、变更记录等）
    一律忽略，因此计划表可以自由增删说明文字。
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到计划表：{path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    title = next((ln.lstrip("# ").strip() for ln in lines if ln.startswith("# ")), path.stem)

    plan = Plan(title=title, source=path)
    current: Module | None = None
    expect_section = False

    for line in lines:
        module_match = MODULE_RE.match(line)
        if module_match:
            code, name = module_match.group(1), module_match.group(2)
            current = Module(code=code, name=name, section="")
            plan.modules.append(current)
            expect_section = True
            continue

        if expect_section:
            expect_section = False  # 只认紧随标题的那一行注释
            section_match = SECTION_RE.match(line)
            if section_match and current is not None:
                current.section = section_match.group(1)
                continue
            # 注意：这里**不能**无条件 continue。标题下面如果没有 section 注释，
            # 当前这一行很可能就是该模块的第一条任务，吞掉它会静默少统计一条。

        task_match = TASK_RE.match(line)
        if task_match and current is not None:
            flag, code, raw_title = task_match.groups()
            deliverable = ""
            deliver_match = DELIVER_RE.search(raw_title)
            if deliver_match:
                deliverable = deliver_match.group(1).strip()
                raw_title = raw_title[: deliver_match.start()].strip()
            current.tasks.append(
                Task(
                    code=code,
                    title=raw_title,
                    status=STATUS_MAP.get(flag, "todo"),
                    deliverable=deliverable,
                )
            )

    return plan


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0d1117; color: #e6edf3; line-height: 1.6;
  font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", -apple-system, sans-serif;
  padding: 32px 24px 64px;
}
.wrap { max-width: 1180px; margin: 0 auto; }
header { margin-bottom: 28px; }
h1 { font-size: 24px; font-weight: 650; letter-spacing: .3px; }
.sub { color: #8b949e; font-size: 13px; margin-top: 6px; }
.headline { display: flex; align-items: baseline; gap: 14px; margin: 22px 0 10px; }
.headline .pct { font-size: 38px; font-weight: 700; color: #58a6ff; font-variant-numeric: tabular-nums; }
.headline .pct small { font-size: 16px; color: #8b949e; font-weight: 500; margin-left: 2px; }
.headline .now { color: #8b949e; font-size: 13px; margin-left: auto; }
.bar { height: 10px; background: #21262d; border-radius: 999px; overflow: hidden; border: 1px solid #30363d; }
.bar > span { display: block; height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, #1f6feb, #58a6ff); transition: width .4s ease; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 12px; margin: 20px 0 34px; }
.stat { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 14px 16px; }
.stat .k { color: #8b949e; font-size: 12px; letter-spacing: .4px; }
.stat .v { font-size: 22px; font-weight: 650; margin-top: 2px; font-variant-numeric: tabular-nums; }
.v.done { color: #3fb950; } .v.doing { color: #d29922; }
.v.blocked { color: #f85149; } .v.todo { color: #8b949e; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(348px, 1fr)); gap: 16px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 18px;
  display: flex; flex-direction: column; }
.card.finished { border-color: #238636; }
.card.active { border-color: #1f6feb; box-shadow: 0 0 0 1px rgba(31,111,235,.35); }
.card.codes { padding: 22px 0; }
.card-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.badge { font-size: 12px; font-weight: 700; padding: 2px 8px; border-radius: 6px;
  background: #1f6feb22; color: #58a6ff; border: 1px solid #1f6feb55; font-family: Consolas, monospace; }
.card.finished .badge { background: #23863622; color: #3fb950; border-color: #23863655; }
.card-title { font-size: 15px; font-weight: 600; }
.tag { font-size: 11px; color: #8b949e; border: 1px solid #30363d; border-radius: 999px;
  padding: 1px 9px; margin-left: auto; white-space: nowrap; }
.card-sub { display: flex; justify-content: space-between; color: #8b949e; font-size: 12px; margin: 12px 0 6px; }
.card .bar { height: 6px; }
.tasks { list-style: none; margin-top: 14px; display: flex; flex-direction: column; gap: 8px; }
.tasks li { display: flex; gap: 9px; font-size: 13px; align-items: flex-start; }
.tasks .ico { flex: 0 0 16px; text-align: center; font-weight: 700; line-height: 1.5; }
.tasks .code { color: #6e7681; font-family: Consolas, monospace; font-size: 12px; flex: 0 0 auto; }
.tasks .txt { flex: 1; }
.tasks .deliver { display: block; color: #6e7681; font-size: 11.5px; margin-top: 2px; }
li.done .ico { color: #3fb950; } li.done .txt { color: #8b949e; text-decoration: line-through;
  text-decoration-color: #484f58; }
li.doing .ico { color: #d29922; } li.doing .txt { color: #e6edf3; font-weight: 600; }
li.blocked .ico { color: #f85149; } li.blocked .txt { color: #f85149; }
li.todo .ico { color: #484f58; } li.todo .txt { color: #c9d1d9; }
footer { margin-top: 40px; color: #6e7681; font-size: 12px; border-top: 1px solid #21262d; padding-top: 16px; }
footer code { background: #161b22; border: 1px solid #30363d; border-radius: 5px; padding: 1px 6px;
  font-family: Consolas, monospace; color: #8b949e; }
"""


def _bar(percent: float, *, extra_class: str = "") -> str:
    return f'<div class="bar {extra_class}"><span style="width:{percent}%"></span></div>'


def _render_module(module: Module, active_code: str | None) -> str:
    card_class = "card"
    if module.is_finished:
        card_class += " finished"
    elif module.code == active_code:
        card_class += " active"

    rows: list[str] = []
    for task in module.tasks:
        deliver = f'<span class="deliver">交付：{html.escape(task.deliverable)}</span>' if task.deliverable else ""
        rows.append(
            f'<li class="{task.status}">'
            f'<span class="ico">{task.icon}</span>'
            f'<span class="code">{html.escape(task.code)}</span>'
            f'<span class="txt">{html.escape(task.title)}{deliver}</span>'
            f"</li>"
        )

    section_tag = f'<span class="tag">{html.escape(module.section)}</span>' if module.section else ""

    return (
        f'<section class="{card_class}">'
        f'<div class="card-head"><span class="badge">{html.escape(module.code)}</span>'
        f'<span class="card-title">{html.escape(module.name)}</span>{section_tag}</div>'
        f'<div class="card-sub"><span>{module.done} / {module.total} 完成</span>'
        f"<span>{module.progress}%</span></div>"
        f"{_bar(module.progress)}"
        f'<ul class="tasks">{"".join(rows)}</ul>'
        f"</section>"
    )


def render_html(plan: Plan, *, generated_at: datetime | None = None) -> str:
    """渲染自包含的深色主题看板（无外部资源依赖，双击即可打开）。"""
    stamp = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M")
    active = plan.current_module
    active_note = f"当前进行：{active.code} {active.name}" if active else "全部模块已完成"

    stats = [
        ("总任务", plan.total, ""),
        ("已完成", plan.done, "done"),
        ("进行中", plan.doing, "doing"),
        ("待办", plan.todo, "todo"),
        ("阻塞", plan.blocked, "blocked"),
    ]
    stat_cards = "".join(
        f'<div class="stat"><div class="k">{name}</div><div class="v {cls}">{value}</div></div>'
        for name, value, cls in stats
    )

    cards = "".join(_render_module(m, active.code if active else None) for m in plan.modules)
    if not plan.modules:
        cards = '<section class="card codes">计划表里还没有模块。</section>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(plan.title)} · 进度看板</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>{html.escape(plan.title)}</h1>
    <div class="sub">数据源：PLAN.md（本看板由 scripts/gen_progress.py 自动生成，请勿直接编辑）</div>
  </header>

  <div class="headline">
    <span class="pct">{plan.progress}<small>%</small></span>
    <span class="sub">{plan.done} / {plan.total} 条子任务完成</span>
    <span class="now">{html.escape(active_note)}</span>
  </div>
  {_bar(plan.progress)}
  <div class="stats">{stat_cards}</div>

  <div class="grid">{cards}</div>

  <footer>
    生成时间：{stamp}　·　更新进度请编辑 <code>PLAN.md</code> 的复选框，然后运行
    <code>python scripts/gen_progress.py</code>
  </footer>
</div>
</body>
</html>
"""


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def generate(plan_path: Path = DEFAULT_PLAN, output_path: Path = DEFAULT_OUTPUT) -> tuple[Path, Plan]:
    """解析计划表并写出看板，返回 (产物路径, 解析结果)。"""
    plan = parse_plan(plan_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(plan), encoding="utf-8")
    return output_path, plan


def format_summary(plan: Plan) -> str:
    """终端摘要文本。"""
    lines = [f"{plan.title}", f"总进度 {plan.progress}%　({plan.done}/{plan.total} 条完成)"]
    for module in plan.modules:
        mark = "✓" if module.is_finished else " "
        lines.append(f"  [{mark}] {module.code} {module.name}　{module.done}/{module.total}　{module.progress}%")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="根据 PLAN.md 生成进度看板")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="计划表路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="看板输出路径")
    parser.add_argument("--print", action="store_true", dest="show", help="在终端打印进度摘要")
    args = parser.parse_args(argv)

    path, plan = generate(args.plan, args.output)
    print(f"进度看板已生成：{path}")
    if args.show:
        print()
        print(format_summary(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
