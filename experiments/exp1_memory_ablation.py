"""实验一：分层记忆消融（有记忆 vs 没记忆，差别有多大）。

对应论文第 6 章实验一，也是整篇论文最核心的一组证据：
**分层记忆到底有没有用。**

## 实验设计

**自变量**是「给模型看多少记忆」，三个条件逐步加料：

    无记忆      什么都不给，直接把问题抛给模型
    仅工作记忆  给最近几轮对话
    全分层记忆  给三层记忆组装出的简报 + 学习者画像

**因变量**是回答里覆盖了几个必备要点。要点由 ``src/eval/judge.py``
让模型来判——「终止条件」和「出口条件」是同一个意思，关键词匹配认不出来。

**为什么条件要这样切**：不是为了凑三档，而是想看清每一层记忆各自
贡献了什么。如果只对比「有 / 无」，就只能说「有用」；
分成三档才能说「工作记忆贡献了哪部分、画像又补上了哪部分」。

## 会花钱

这个实验要真实调用模型：3 个条件 × 3 个问题 = 9 次生成，
再加 9 次要点判定，一共 18 次。跑一次约两三分钟。

运行::

    .venv\\Scripts\\python.exe experiments\\exp1_memory_ablation.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import (  # noqa: E402
    MEMORY_CONDITIONS,
    PROFILE_QUESTIONS,
    STUDENT_SESSION,
    coverage,
    mean,
    std,
)
from src.eval.judge import judge_points  # noqa: E402
from src.llm import create_chat_model  # noqa: E402
from src.memory import MemoryManager  # noqa: E402
from src.planning.profile import LearnerProfile  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = PROJECT_ROOT / "docs" / "experiments"

ASK_TEMPLATE = """你是一位学习助手。请根据下面提供的信息回答学生的问题。

{context}

学生的问题：{question}

要求：直接回答这个问题，不要复述上面给的信息，也不要反问学生。"""

NO_CONTEXT = "（没有关于这位学生的任何记录）"


def seed(memory: MemoryManager, profile: LearnerProfile) -> None:
    """把数据集里的学生轨迹灌进记忆与画像。

    三个条件共用同一份数据，差别只在于「给模型看多少」。
    灌数据这一步完全一致，才能保证结果的差异来自条件本身。
    """
    for role, content in STUDENT_SESSION["dialogue"]:
        memory.add_message(role, content, learn=False)

    for event in STUDENT_SESSION["events"]:
        memory.record_event(event["event_type"], event["content"], event["importance"])

    for topic, mastery in STUDENT_SESSION["mastery"].items():
        profile.set_mastery(topic, mastery)

    profile.set_goal(STUDENT_SESSION["goal"])


def build_context(
    condition: dict,
    memory: MemoryManager,
    profile: LearnerProfile,
) -> str:
    """按条件拼出给模型看的上下文。

    「仅工作记忆」那一档要单独拼，不能直接调 ``memory.context()``——
    后者会把三层记忆全带上，条件就不成立了。这种「不小心多给了信息」
    的错误不会报错，只会让结论看起来更好看，所以必须写清楚。
    """
    parts: list[str] = []

    if condition["use_working"]:
        messages = memory.working.messages()
        if messages:
            lines = ["【最近对话】"]
            lines += [f"- {item['role']}：{item['content']}" for item in messages]
            parts.append("\n".join(lines))

    if condition["use_episodic"]:
        events = memory.episodic.recent(10)
        if events:
            lines = ["【最近发生的事】"]
            lines += [f"- {item['content']}" for item in events]
            parts.append("\n".join(lines))

    if condition["use_semantic"]:
        facts = memory.semantic.all_facts()
        if facts:
            lines = ["【沉淀的事实】"]
            lines += [f"- {item['key']}：{item['value']}" for item in facts]
            parts.append("\n".join(lines))

    if condition["use_profile"]:
        snapshot = profile.snapshot()
        if snapshot.strip():
            parts.append(f"【学习者画像】\n{snapshot}")

    return "\n\n".join(parts) if parts else NO_CONTEXT


def run_condition(condition: dict, model, db_path: Path) -> dict:
    """跑一个实验条件。"""
    memory = MemoryManager(
        session_id=f"{STUDENT_SESSION['session_id']}-{condition['key']}",
        user_id=STUDENT_SESSION["user_id"],
        db_path=db_path,
    )
    profile = LearnerProfile(STUDENT_SESSION["user_id"], db_path=db_path)
    profile.clear()
    seed(memory, profile)

    context = build_context(condition, memory, profile)

    rows: list[dict] = []
    for item in PROFILE_QUESTIONS:
        prompt = ASK_TEMPLATE.format(context=context, question=item["question"])
        reply = model.invoke(prompt)
        answer = reply.content if hasattr(reply, "content") else str(reply)

        marks = judge_points(item["question"], answer, item["points"], model=model)

        rows.append(
            {
                "question": item["question"],
                "points": item["points"],
                "marks": marks,
                "coverage": coverage(marks),
                "answer": answer,
            }
        )

    profile.close()
    memory.close()

    return {
        "key": condition["key"],
        "label": condition["label"],
        "context_chars": len(context),
        "overall": round(mean([row["coverage"] for row in rows]), 4),
        "rows": rows,
    }


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="exp1-"))
    db_path = workdir / "ablation.db"
    model = create_chat_model(temperature=0.2)

    results: list[dict] = []

    try:
        for condition in MEMORY_CONDITIONS:
            print(f"  跑 {condition['label']} ……", flush=True)
            results.append(run_condition(condition, model, db_path))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    print("=" * 84)
    print("实验一　分层记忆消融　（要点覆盖率）")
    print("=" * 84)
    print(f"问题数：{len(PROFILE_QUESTIONS)}　每题要点数：{len(PROFILE_QUESTIONS[0]['points'])}")
    print()

    print(f"{'条件':<12}{'上下文长度':>12}{'要点覆盖率':>12}")
    print("-" * 84)

    for item in results:
        print(f"{item['label']:<12}{item['context_chars']:>10} 字{item['overall']:>12.4f}")

    print()
    print("逐题明细：")
    for index, question in enumerate(PROFILE_QUESTIONS):
        print(f"  Q{index + 1}. {question['question']}")
        for item in results:
            row = item["rows"][index]
            marks = "".join("✓" if mark else "✗" for mark in row["marks"])
            print(f"      {item['label']:<10} {marks}  覆盖率 {row['coverage']:.2f}")

    print()
    print("结论：")
    baseline = results[0]["overall"]
    full = results[-1]["overall"]
    for item in results[1:]:
        delta = item["overall"] - baseline
        print(
            f"  {item['label']:<10} 相比无记忆 {delta:+.4f}"
            f"（相对提升 {(delta / baseline) if baseline else 0:+.0%}）"
        )
    print(
        f"  全分层记忆相比无记忆提了 {full - baseline:.4f}，"
        "这就是「分层记忆」在回答质量上的直接体现。"
    )
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "exp1_ablation.json"
    out_file.write_text(
        json.dumps(
            {
                "config": {
                    "conditions": [item["key"] for item in MEMORY_CONDITIONS],
                    "questions": len(PROFILE_QUESTIONS),
                },
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"结果已写入 {out_file.relative_to(PROJECT_ROOT)}")
    print("（这个实验会真实调用模型，重跑前留意费用）")


if __name__ == "__main__":
    main()
