"""学习闭环 demo：完整跑一遍「学 → 练 → 评 → 调」。

对应计划 M4.7，也是论文第 5 章的可演示成果。

这个 demo 会把整条链路走一遍：

    拆解目标    「掌握递归」→ 拆成几个带依赖关系的知识点
    学          导师 Agent 针对这个学生讲一段
    练          出题 Agent 出几道题
    评          评估 Agent 批改（学生由脚本模拟，答案预先写好）
    调          按批改结果更新掌握度，重新排路径，必要时回退补前置

**注意**：这个脚本会真实调用 DeepSeek，大约 10~15 次。想省费用可以把
``MAX_ROUNDS`` 调小，或者把 ``GOAL`` 换成更小的目标。

运行::

    .venv\\Scripts\\python.exe examples\\demo_learning.py

数据存在 ``data/demo_learning.db``（已在 .gitignore 中排除），每次运行前会清空。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让脚本能直接跑：把项目根目录加进模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.planning.graph import LearningSession  # noqa: E402
from src.planning.planner import STATUS_FOCUS, STATUS_PRACTICE, STATUS_REVIEW  # noqa: E402

# Windows 控制台默认 GBK，打印中文可能报编码错，统一成 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 学习目标。拆解粒度由 max_topics 控制，不想花钱就调小。
GOAL = "掌握递归"

# 一次最多批改几轮，防止 demo 跑太久
MAX_ROUNDS = 8

DEMO_DB = PROJECT_ROOT / "data" / "demo_learning.db"

# 模拟学生的作答。写得有对有错，才能看出「重讲」和「换知识点」都被触发。
# 真实场景里这些文字来自屏幕那头的人。
SIMULATED_ANSWERS = [
    "就是函数自己调用自己吧，具体怎么写没太想清楚",
    "递归要有一个终止条件让它停下来，还要把问题拆成更小的同类问题，一直拆到能直接算出来为止",
    "终止条件写成 n <= 1 的时候直接返回，剩下的交给下一次调用去算",
    "有点绕，我不太确定怎么保证不会无限递归下去",
    "先写终止条件，再写递归那一步，递归那步必须让问题变小",
]

_STATUS_LABEL = {
    STATUS_FOCUS: "重点攻",
    STATUS_PRACTICE: "多练习",
    STATUS_REVIEW: "快速过",
}


def rule(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def show_path(session: LearningSession) -> None:
    """打印当前路径上每个知识点的状态。

    除了掌握程度，还标出「本次已跳过」和「待回头补」——
    这两类状态正是动态重规划留下的痕迹，光看掌握度看不出来。
    """
    steps = session.state.get("path", {}).get("steps", [])
    if not steps:
        return

    skipped = set(session.state.get("skipped", []))
    needs_review = set(session.state.get("needs_review", []))

    print("  当前路径：")
    for step in steps:
        topic = step["topic"]
        marks = []
        if topic == session.state.get("topic"):
            marks.append("正在学")
        if topic in skipped:
            marks.append("本次已跳过")
        if topic in needs_review:
            marks.append("待回头补")

        suffix = f"　← {'、'.join(marks)}" if marks else ""
        print(
            f"    {step['order']}. {topic}　[{_STATUS_LABEL[step['status']]}]"
            f"　掌握度 {step['mastery']:.2f}{suffix}"
        )


def main() -> None:
    # 每次运行都用干净的库，多跑几次结果可比
    DEMO_DB.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        Path(str(DEMO_DB) + suffix).unlink(missing_ok=True)

    session = LearningSession(
        goal=GOAL,
        user_id="demo-student",
        session_id="demo-session",
        db_path=DEMO_DB,
        max_topics=3,
    )

    try:
        rule("第 1 步　拆解目标：把「" + GOAL + "」拆成知识点")
        state = session.start()
        for item in state["topics"]:
            deps = "、".join(item["depends_on"]) or "无"
            print(f"  · {item['topic']}")
            print(f"      前置：{deps}")
            if item.get("reason"):
                print(f"      理由：{item['reason']}")

        rule("第 2 步　导师讲解 + 出题")
        show_path(session)
        print()
        print(f"  【导师讲解《{state['topic']}》】")
        print("  " + state["explanation"].replace("\n", "\n  "))

        question = session.current_question()
        print()
        print(f"  【第 1 题】{question['question']}")

        rule("第 3 步　批改与调整：学生逐题作答，系统边评边调")

        for round_index in range(MAX_ROUNDS):
            question = session.current_question()
            if question is None or session.finished:
                break

            answer = SIMULATED_ANSWERS[round_index % len(SIMULATED_ANSWERS)]
            topic_before = session.state["topic"]

            state = session.answer(answer)

            grade = state.get("grade", {})
            verdict = "✓ 对" if grade.get("correct") else "✗ 错"
            print()
            print(f"  ── 第 {round_index + 1} 轮 ──")
            print(f"  题目：{question['question']}")
            print(f"  学生：{answer}")
            print(f"  判定：{verdict}　{grade.get('feedback', '')}")

            topic_after = state.get("topic", "")
            if session.finished:
                print("  → 本次学习结束")
            elif topic_after != topic_before:
                print(f"  → 换知识点：《{topic_before}》→《{topic_after}》")
            elif state.get("retries", 0) > 0:
                print(f"  → 答错了，换个讲法重讲《{topic_before}》")
                print(f"    新讲解：{state['explanation'][:60]}…")

            if state.get("needs_review"):
                print(f"  → 待补的前置：{'、'.join(state['needs_review'])}")

        rule("第 4 步　收尾：这次学到了什么")
        show_path(session)
        print()
        print("  最终画像：")
        print("  " + session.profile.snapshot().replace("\n", "\n  "))
        print()
        print(f"  记忆里留下了 {session.memory.stats()['events']} 条事件、"
              f"{session.memory.stats()['messages']} 条对话、"
              f"{session.memory.stats()['facts']} 条事实")
        print()
        print("  过程日志：")
        for item in session.state.get("log", []):
            print(f"    · {item['entry']}")

        print()
        print("循环次数受 MAX_ROUNDS 限制，跑到一半是正常的——")
        print("真实使用时学生可以随时接着来，状态都在数据库里。")

    finally:
        session.close()


if __name__ == "__main__":
    main()
