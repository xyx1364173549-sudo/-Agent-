"""分层记忆最小可运行 demo。

跑起来能看到什么？

    同一句提问「我最近在学什么？薄弱点在哪？」，问同一个模型两次：
      - 情况 A：不带任何记忆 —— 模型只能瞎猜，或者说不知道
      - 情况 B：带上分层记忆 —— 模型准确说出你在学递归、卡在终止条件

差别就是「分层记忆」的价值所在，也是论文实验一的雏形。

运行方式::

    python examples/demo_memory.py

数据存在 data/demo.db（已在 .gitignore 中排除），每次运行前会清空，
所以多跑几次结果一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让脚本能直接跑：把项目根目录加进模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import create_chat_model  # noqa: E402
from src.memory import EpisodicMemory, SemanticMemory, WorkingMemory, count_tokens  # noqa: E402

# Windows 控制台默认 GBK，打印中文可能报编码错，统一成 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SESSION_ID = "demo-session"
USER_ID = "demo-user"
DEMO_DB = PROJECT_ROOT / "data" / "demo.db"

# 模拟的一段对话：一个正在学递归、卡在终止条件的学生
DIALOGUE = [
    ("user", "我最近在学递归，感觉有点难"),
    ("assistant", "递归确实不容易，关键是想清楚终止条件"),
    ("user", "我练习了 5 道题，答错了 3 道"),
    ("assistant", "那我们把重点放在终止条件上"),
    ("user", "递归我还是不太会"),
    ("assistant", "别急，我们一步步来"),
]

# demo 最后要问模型的问题
QUESTION = "根据你对我的了解，我最近在学什么？薄弱点在哪？"


def build_context(
    working: WorkingMemory,
    episodic: EpisodicMemory,
    semantic: SemanticMemory,
    *,
    max_tokens: int = 600,
) -> str:
    """把三层记忆拼成一段能直接塞给模型的文字。

    拼的顺序有讲究：**越稳定、越浓缩的记忆放越前面**。

    1. 语义记忆 —— 已经沉淀的事实，最可靠，而且很短
    2. 情景记忆 —— 最近发生的事，用来补时间线
    3. 工作记忆 —— 最近的原话，最细但最占地方

    这样即使后面 token 超了要砍，砍掉的也是最不重要的原文，
    留下的仍是「这个人是什么水平」这个核心判断。
    """
    parts: list[str] = []

    facts = semantic.all_facts()
    if facts:
        parts.append("【学习者情况】")
        parts.extend(f"- {f['key']}：{f['value']}" for f in facts)

    events = episodic.recent(5)
    if events:
        parts.append("【最近发生的事】")
        parts.extend(f"- {e['created_at'][11:16]} {e['content']}" for e in events)

    messages = working.messages(max_tokens=max_tokens)
    if messages:
        parts.append("【最近对话】")
        parts.extend(f"- {m['role']}：{m['content']}" for m in messages)

    return "\n".join(parts)


def main() -> int:
    line = "=" * 62
    print(line)
    print(" 分层记忆 Demo：让 Agent 记住你是谁")
    print(line)

    working = WorkingMemory(SESSION_ID, db_path=DEMO_DB)
    episodic = EpisodicMemory(SESSION_ID, db_path=DEMO_DB)
    semantic = SemanticMemory(USER_ID, db_path=DEMO_DB)

    # 每次运行前清空，保证 demo 结果可复现
    working.clear()
    episodic.clear()
    semantic.clear()

    # ---------------------------------------------------------------- 第 1 步
    print("\n【第 1 步】模拟一段对话，写进三层记忆")
    print("-" * 62)
    for role, text in DIALOGUE:
        working.add(role, text)
        print(f"  工作记忆 ← {role}：{text}")

        # 用户说的话，顺手抽出事实存进语义记忆
        if role == "user":
            for item in semantic.learn_from_text(text, confidence=0.7):
                action = {"created": "新增", "reinforced": "再次确认", "updated": "更新", "kept": "保留旧结论"}
                print(f"      语义记忆 ← {item['key']} = {item['value']}（{action.get(item['action'], item['action'])}）")

    episodic.record("struggled", "练习递归 5 题，答错 3 题", importance=0.8)
    print("  情景记忆 ← 练习递归 5 题，答错 3 题（重要性 0.8）")

    # ---------------------------------------------------------------- 第 2 步
    print("\n【第 2 步】三层记忆里现在存了什么")
    print("-" * 62)

    print("  语义记忆（档案卡片，沉淀的事实）")
    for fact in semantic.all_facts():
        print(f"    - {fact['key']}：{fact['value']}（置信度 {fact['confidence']:.1f}）")
    if not semantic.all_facts():
        print("    （空）")

    print("  情景记忆（日记，发生过的事）")
    for event in episodic.recent(5):
        print(f"    - {event['created_at'][11:16]}  {event['content']}")

    kept = working.messages()
    print(f"  工作记忆（草稿纸，共 {working.count()} 条，取出 {len(kept)} 条 / 约 {sum(count_tokens(m['content']) for m in kept)} token）")

    context = build_context(working, episodic, semantic)
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
        return 1

    print("  ── 情况 A：不带任何记忆 ──")
    answer_a = model.invoke(QUESTION).content
    print(f"  {answer_a}\n")

    print("  ── 情况 B：带上分层记忆 ──")
    answer_b = model.invoke(f"{context}\n\n{QUESTION}").content
    print(f"  {answer_b}\n")

    print(line)
    print(" Demo 结束。两种情况差别明显吗？")
    print(" 差别越大，说明记忆起的作用越明显 —— 这正是论文实验一要量化的东西。")
    print(line)

    working.close()
    episodic.close()
    semantic.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
