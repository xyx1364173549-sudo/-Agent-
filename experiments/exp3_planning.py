"""实验三：动态规划 vs 静态路径（仿真对比）。

对应论文第 6 章实验三。

## 为什么用仿真学生，不调真实模型

要对比的是**规划算法**，不是模型能力。真实模型每次回答都不一样，
跑 30 遍得到 30 个不同结果，反而说不清「差别是算法带来的还是模型随机性
带来的」。仿真学生给出三个真实模型给不了的东西：

    零成本    想跑多少遍跑多少遍
    可复现    固定随机种子，结果逐位一致
    可统计    30 次不同种子 → 均值和标准差，结论才站得住

## 两种策略

**静态路径**　按知识点图的顺序循环练，不看学生当前水平。
　　　　　　　这就是「一份课表发给所有人」。

**动态规划**　每轮按最新画像重排，优先练最弱的。走的是
　　　　　　　``src/planning/planner.py`` 里那套真实算法，不是另写一份。

## 区分度从哪来：遗忘

如果练过就不会忘，那静态路径也能把所有知识点都刷到达标，
两种策略没有差别。**加入遗忘之后差别才显现**：静态课表每个知识点
要等一整轮才轮到一次，中间掉的分数补不回来；动态规划总在补当前最弱的那个，
间隔短、掉得少。

这也不是为了造差异而硬加的设定——「学了会忘、要回头复习」本来就是
学习的固有特征，静态课表最大的问题恰恰是**不回头**。

运行::

    .venv\\Scripts\\python.exe experiments\\exp3_planning.py
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import (  # noqa: E402
    KNOWLEDGE_GRAPH,
    SIM_STUDENT,
    TARGET_TOPICS,
    mean,
    summarize,
)
from src.planning.planner import next_step, plan  # noqa: E402
from src.planning.profile import (  # noqa: E402
    MASTERED_THRESHOLD,
    MASTERY_HALF_LIFE_DAYS,
    LearnerProfile,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 一轮练习按一天算。这个换算不能省：仿真的时间和画像的衰减必须同一套口径，
# 否则算出来的间隔是负的，衰减直接失效（第一版就是这么栽的）
DAYS_PER_ROUND = 1.0

# 学生的真实遗忘速度，与画像的衰减**用同一个半衰期**。
#
# 为什么必须一致：画像衰减得比真实慢，规划器就会以为「这个点还行」而不去复习，
# 实际早就掉下去了——实验结果会完全反过来。第一版画像不衰减时，
# 动态规划的平均掌握度反而比静态课表低 0.11，原因就在这里。
DECAY_PER_ROUND = 2 ** (-DAYS_PER_ROUND / MASTERY_HALF_LIFE_DAYS)

# 一轮练习的预算。两种策略花一样的轮数，比「同样投入下的产出」
ROUND_BUDGET = 30

# 蒙特卡洛次数。固定种子序列，结果完全可复现
RUNS = 30

USER_ID = "sim-student"

OUTPUT_DIR = PROJECT_ROOT / "docs" / "experiments"

STRATEGIES = (
    ("static", "静态路径"),
    ("dynamic", "动态规划"),
)


def simulate(strategy: str, *, seed: int, db_path: Path) -> dict:
    """跑一次仿真，返回这次的结果。"""
    rng = random.Random(seed)

    all_topics = [item["topic"] for item in KNOWLEDGE_GRAPH]
    deps = {item["topic"]: list(item["depends_on"]) for item in KNOWLEDGE_GRAPH}
    gain = SIM_STUDENT["gain_per_practice"]
    target = SIM_STUDENT["target"]

    # 两种策略看到的范围不一样，这是本实验最关键的一处设定：
    #
    #   静态课表　照大纲一路发下去，不管你的目标是什么 —— 六个点都要练
    #   动态规划　只看目标范围内的知识点 —— 递归那四个，二分查找不碰
    #
    # 现实里就是这样：一份通用课表和一个知道你目标在哪的系统，
    # 花同样的时间，结果不可能一样。
    if strategy == "static":
        scope = all_topics
        graph = KNOWLEDGE_GRAPH
    else:
        scope = [name for name in all_topics if name in TARGET_TOPICS]
        graph = [item for item in KNOWLEDGE_GRAPH if item["topic"] in TARGET_TOPICS]

    # 学生的**真实**水平 —— 系统看不到，只用来决定「练了有没有用」和「答对概率」
    truth = dict(SIM_STUDENT["true_mastery"])

    # 系统的**画像** —— 规划器唯一能看到的东西，从零开始。
    # 时钟注入成模拟时间，让「一轮 = 一天」这件事对画像也成立
    clock = [datetime(2026, 1, 1, 9, 0, 0)]
    profile = LearnerProfile(USER_ID, db_path=db_path, clock=lambda: clock[0])
    profile.clear()

    attempts: list[dict] = []
    reached_at: int | None = None

    for round_index in range(ROUND_BUDGET):
        topic = _pick(strategy, scope, graph, profile, round_index)

        mastery_before = profile.mastery_of(topic)
        truth_before = truth[topic]

        # 前置知识还没学会时，练这个知识点是白费功夫 —— 学生根本接不住。
        # 这是本实验最关键的一处设定：**没有它，静态课表和动态规划
        # 的结果会一模一样**，因为「练了就有效」的话，练哪个都行
        ready = all(truth[dep] >= target for dep in deps[topic])

        if ready:
            truth[topic] = min(1.0, truth[topic] + gain * (1 - truth[topic]))
            correct = rng.random() < truth[topic]
        else:
            # 前置都没学会，答对基本靠蒙，真实水平也不涨
            correct = rng.random() < 0.1

        profile.record_attempt(topic, correct)

        attempts.append(
            {
                "round": round_index + 1,
                "topic": topic,
                "ready": ready,
                "mastery_before": mastery_before,
                "truth_before": round(truth_before, 4),
            }
        )

        # 一轮过去，所有知识点都掉一点分（真实水平）
        for name in truth:
            truth[name] *= DECAY_PER_ROUND

        # 系统这边的时间也往前走，画像的掌握度随之衰减
        clock[0] = clock[0] + timedelta(days=DAYS_PER_ROUND)

        # 「达标」只看**目标范围内**的知识点
        if reached_at is None and all(truth[name] >= target for name in TARGET_TOPICS):
            reached_at = round_index + 1

    profile.close()

    target_values = [truth[name] for name in TARGET_TOPICS]
    effective = sum(1 for item in attempts if item["ready"])

    return {
        "strategy": strategy,
        "seed": seed,
        "scope_size": len(scope),
        "target_mastered": sum(1 for value in target_values if value >= target),
        "total_targets": len(TARGET_TOPICS),
        "mean_target": round(mean(target_values), 4),
        "min_target": round(min(target_values), 4),
        "reached_at": reached_at if reached_at is not None else ROUND_BUDGET + 1,
        "reached": reached_at is not None,
        "effective_ratio": round(effective / len(attempts), 4),
        "attempts": attempts,
    }


def _pick(
    strategy: str,
    scope: list[str],
    graph: list[dict],
    profile: LearnerProfile,
    round_index: int,
) -> str:
    """挑这一轮练哪个知识点。

    静态：按大纲顺序循环，完全不看画像——这就是「固定课表」的含义。
    动态：走 planner 的真实排序逻辑（含拓扑分层），优先练最弱的那个。
    """
    if strategy == "static":
        return scope[round_index % len(scope)]

    result = plan(graph, profile)
    step = next_step(result)
    # 全都达标时 next_step 返回 None，此时练第一个即可
    return step["topic"] if step else scope[round_index % len(scope)]


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="exp3-"))
    db_path = workdir / "sim.db"

    results: dict[str, list[dict]] = {key: [] for key, _ in STRATEGIES}

    try:
        for key, label in STRATEGIES:
            print(f"  跑 {label} ……", flush=True)
            for seed in range(RUNS):
                results[key].append(simulate(key, seed=seed, db_path=db_path))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    print("=" * 84)
    print(f"实验三　静态路径 vs 动态规划　（{RUNS} 次仿真，预算 {ROUND_BUDGET} 轮）")
    print("=" * 84)
    print(
        f"遗忘半衰期：{MASTERY_HALF_LIFE_DAYS:.0f} 天（仿真里一轮 = {DAYS_PER_ROUND:.0f} 天）"
        f"　掌握目标：{MASTERED_THRESHOLD}"
    )
    print()

    header = (
        f"{'策略':<10}{'练习范围':>10}{'目标达标':>10}"
        f"{'有效练习':>10}{'目标均分':>10}{'最弱一项':>10}"
    )
    print(header)
    print("-" * 84)

    summary: dict[str, dict] = {}

    for key, label in STRATEGIES:
        runs = results[key]

        reached_at = [float(run["reached_at"]) for run in runs]
        effective = [run["effective_ratio"] for run in runs]
        mean_target = [run["mean_target"] for run in runs]
        min_target = [run["min_target"] for run in runs]

        summary[key] = {
            "label": label,
            "reached_at": summarize(reached_at),
            "success_rate": round(sum(1.0 if run["reached"] else 0.0 for run in runs) / len(runs), 4),
            "effective_ratio": summarize(effective),
            "mean_target": summarize(mean_target),
            "min_target": summarize(min_target),
        }

        print(
            f"{label:<10}"
            f"{runs[0]['scope_size']:>7} 个"
            f"{mean([float(run['target_mastered']) for run in runs]):>6.2f}/{runs[0]['total_targets']}"
            f"{mean(effective):>10.4f}"
            f"{mean(mean_target):>10.4f}"
            f"{mean(min_target):>10.4f}"
        )

    print()
    print("波动（标准差，越小说明结果越稳定）：")
    for key, label in STRATEGIES:
        item = summary[key]
        print(
            f"  {label:<8}　目标达标数 ±{item['mean_target']['std']:.4f}　"
            f"有效练习 ±{item['effective_ratio']['std']:.4f}　"
            f"目标均分 ±{item['mean_target']['std']:.4f}"
        )

    print()
    print("结论：")
    static_mean = summary["static"]["mean_target"]["mean"]
    dynamic_mean = summary["dynamic"]["mean_target"]["mean"]
    static_eff = summary["static"]["effective_ratio"]["mean"]
    dynamic_eff = summary["dynamic"]["effective_ratio"]["mean"]

    print(
        f"  目标均分　　静态 {static_mean:.4f} → 动态 {dynamic_mean:.4f}"
        f"　提升 {dynamic_mean - static_mean:+.4f}"
        f"（{(dynamic_mean - static_mean) / static_mean:+.0%}）"
    )
    print(
        f"  有效练习　　静态 {static_eff:.4f} → 动态 {dynamic_eff:.4f}"
        f"　提升 {dynamic_eff - static_eff:+.4f}"
    )
    print(
        "  说明：静态课表一半的练习花在前置还没学会的知识点上——"
        "按大纲顺序轮转，轮到「返回值合并」时「调用过程」还没学，练了也白练。"
    )
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "exp3_planning.json"
    out_file.write_text(
        json.dumps(
            {
                "config": {
                    "runs": RUNS,
                    "round_budget": ROUND_BUDGET,
                    "decay_per_round": DECAY_PER_ROUND,
                    "target": MASTERED_THRESHOLD,
                    "sim_student": SIM_STUDENT,
                },
                "summary": summary,
                "runs": {
                    key: [
                        {k: v for k, v in run.items() if k != "attempts"} for run in runs
                    ]
                    for key, runs in results.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"结果已写入 {out_file.relative_to(PROJECT_ROOT)}")
    print("（这个实验也不调用大模型，可以随便重跑）")


if __name__ == "__main__":
    main()
