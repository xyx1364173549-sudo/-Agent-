"""分层记忆最小可运行 demo。

跑起来能看到什么？

    同一句提问「我最近在学什么？薄弱点在哪？」，问同一个模型两次：
      · 情况 A：不带任何记忆 —— 模型只能瞎猜，或者说不知道
      · 情况 B：带上分层记忆 —— 模型准确说出你在学递归、卡在终止条件

    最后再演示一次「时间快进」：哪些记忆会被忘掉，哪些能留下来。

运行方式::

    python examples/demo_memory.py

数据存在 data/demo.db（已在 .gitignore 中排除），每次运行前会清空，
所以多跑几次结果一致。
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

# 让脚本能直接跑：把项目根目录加进模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import create_chat_model  # noqa: E402
from src.memory import MemoryManager, count_tokens, strength  # noqa: E402

# Windows 控制台默认 GBK，打印中文可能报编码错，统一成 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SESSION_ID = "demo-session"
USER_ID = "demo-user"
DEMO_DB = PROJECT_ROOT / "data" / "demo.db"

LINE = "=" * 62

# 模拟的一段对话：一个正在学递归、卡在终止条件的学生
DIALOGUE = [
    ("user", "我最近在学递归，感觉有点难"),
    ("assistant", "递归确实不容易，关键是想清楚终止条件"),
    ("user", "我练习了 5 道题，答错了 3 道"),
    ("assistant", "那我们把重点放在终止条件上"),
    ("user", "递归我还是不太会"),
    ("assistant", "别急，我们一步步来"),
]

# 从对话里沉淀出来的事件，带各自的重要程度
EVENTS = [
    ("chitchat", "聊了句今天天气不错", 0.2),
    ("struggled", "练习递归 5 题，答错 3 题", 0.8),
]

# demo 最后要问模型的问题
QUESTION = "根据你对我的了解，我最近在学什么？薄弱点在哪？"

# 遗忘曲线要看的几个时间点（天）
MILESTONES = (0, 7, 14, 30)

# 归档阈值：强度低于它就算「想不起来了」
FORGET_THRESHOLD = 0.2


def main() -> int:
    print(LINE)
    print(" 分层记忆 Demo：让 Agent 记住你是谁")
    print(LINE)

    mem = MemoryManager(SESSION_ID, USER_ID, db_path=DEMO_DB)

    # 每次运行前清空，保证 demo 结果可复现
    mem.clear()

    # ---------------------------------------------------------------- 第 1 步
    print("\n【第 1 步】模拟一段对话，写进三层记忆")
    print("-" * 62)
    action_names = {"created": "新增", "reinforced": "再次确认", "updated": "更新", "kept": "保留旧结论"}

    for role, text in DIALOGUE:
        learned = mem.add_message(role, text)
        print(f"  工作记忆 ← {role}：{text}")
        for item in learned:
            print(f"      语义记忆 ← {item['key']} = {item['value']}（{action_names.get(item['action'], item['action'])}）")

    for event_type, content, importance in EVENTS:
        mem.record_event(event_type, content, importance=importance)
        print(f"  情景记忆 ← {content}（重要性 {importance}）")

    # ---------------------------------------------------------------- 第 2 步
    print("\n【第 2 步】三层记忆里现在存了什么")
    print("-" * 62)

    print("  语义记忆（档案卡片，沉淀的事实）")
    for fact in mem.semantic.all_facts():
        print(f"    - {fact['key']}：{fact['value']}（置信度 {fact['confidence']:.1f}）")

    print("  情景记忆（日记，发生过的事）")
    for event in mem.episodic.recent(10):
        print(f"    - {event['created_at'][11:16]}  {event['content']}（重要性 {event['importance']:.1f}）")

    messages = mem.working.messages()
    tokens = sum(count_tokens(m["content"]) for m in messages)
    print(f"  工作记忆（草稿纸，共 {mem.working.count()} 条，取用 {len(messages)} 条 / 约 {tokens} token）")

    context = mem.context()
    print("\n  组装成给模型的上下文：")
    for text in context.splitlines():
        print(f"    | {text}")

    # ---------------------------------------------------------------- 第 3 步
    print("\n【第 3 步】同一句提问，两种情况下模型的回答")
    print("-" * 62)
    print(f"  提问：{QUESTION}\n")

    try:
        model = create_chat_model(temperature=0.3)
    except RuntimeError as exc:
        print(f"  [跳过] 没法调用模型：{exc}")
        print("  提示：请在项目根目录建 .env 文件，写入 DEEPSEEK_API_KEY=sk-xxxx")
        mem.close()
        return 1

    print("  ── 情况 A：不带任何记忆 ──")
    print(f"  {model.invoke(QUESTION).content}\n")

    print("  ── 情况 B：带上分层记忆 ──")
    print(f"  {model.invoke(f'{context}\n\n{QUESTION}').content}\n")

    # ---------------------------------------------------------------- 第 4 步
    print("\n【第 4 步】时间快进：记忆会怎样淡忘")
    print("-" * 62)
    print(f"  半衰期 7 天；强度低于 {FORGET_THRESHOLD} 就算想不起来了\n")

    header = f"  {'事件':<26}{'重要性':>7}" + "".join(f"{d}天后".rjust(9) for d in MILESTONES)
    print(header)
    for event in mem.episodic.recent(10):
        cells = []
        for days in MILESTONES:
            moment = datetime.now() + timedelta(days=days)
            cells.append(f"{strength(event['importance'], event['created_at'], now=moment):>9.2f}")
        label = event["content"][:24]
        print(f"  {label:<26}{event['importance']:>7.1f}" + "".join(cells))

    # 选 7 天这个时间点：不重要的那条正好掉到阈值以下，重要的那条还挺得住，
    # 差别才看得出来。拨到 30 天的话两条都忘光了，反而看不出「重要的事留得久」。
    future = datetime.now() + timedelta(days=7)
    archived = mem.episodic.forget_weak(threshold=FORGET_THRESHOLD, now=future)
    print(f"\n  把时间拨到 7 天后 → {archived} 条被遗忘")
    print(f"  还记着的：{[e['content'] for e in mem.episodic.recent(10)] or '（空）'}")
    print(f"  已归档的：{mem.episodic.archived_count()} 条（数据还在库里，只是想不起来了）")
    print("  注意：重要的事留下来了，不重要的事被忘掉——这就是「重要性越高越抗忘」。")

    print("\n" + LINE)
    print(" 两种情况差别明显吗？差别越大，说明记忆起的作用越明显。")
    print(" 这正是论文实验一要量化的东西。")
    print(LINE)

    mem.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
