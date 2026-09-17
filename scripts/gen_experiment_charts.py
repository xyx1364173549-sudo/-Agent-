"""实验图表生成：把三个实验的 JSON 结果画成论文可直接引用的图。

为什么单独一个脚本而不是塞进实验脚本里：

    实验脚本只管「跑实验、出数据」——数据存在 JSON 里之后，想调整图表
    样式、换配色、加一张图，都不用重新跑实验（实验一要调模型的，
    重跑就是再花一遍钱）。
    图表脚本的输入是结果文件，跑多少次都不要钱。

输出：docs/experiments/charts/*.png，300 dpi，白底——论文插图的标准要求。

运行::

    .venv\\Scripts\\python.exe scripts/gen_experiment_charts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS_DIR = PROJECT_ROOT / "docs" / "experiments"
CHARTS_DIR = EXPERIMENTS_DIR / "charts"

# 论文插图用白底 + 深色文字，与正文风格一致
plt_style = {"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"}


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Windows 自带的中文字体，避免图表里的中文变成方块
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update(plt_style)
    return plt


def chart_exp1() -> None:
    """实验一：三个记忆条件的要点覆盖率对比。"""
    data = json.loads((EXPERIMENTS_DIR / "exp1_ablation.json").read_text(encoding="utf-8"))
    results = data["results"]

    plt = _setup_matplotlib()

    labels = [item["label"] for item in results]
    values = [item["overall"] for item in results]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#9aa0a6", "#5b8def", "#3fb950"]
    bars = ax.bar(labels, values, color=colors, width=0.55)

    # 每个柱子顶上标数值，读者不用对着 y 轴读数
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylabel("要点覆盖率", fontsize=11)
    ax.set_title("实验一　分层记忆消融：不同条件下的回答质量", fontsize=13, pad=12)
    ax.set_ylim(0, 1.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / "exp1_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  exp1_ablation.png")


def chart_exp2() -> None:
    """实验二：切分策略 × 检索策略的 Hit@2 对比。"""
    data = json.loads((EXPERIMENTS_DIR / "exp2_retrieval.json").read_text(encoding="utf-8"))
    results = data["results"]

    plt = _setup_matplotlib()

    strategies = ("fixed", "recursive", "semantic")
    modes = ("keyword", "vector", "hybrid")
    mode_labels = {"keyword": "关键词", "vector": "向量", "hybrid": "混合"}
    strategy_labels = {"fixed": "固定长度", "recursive": "递归", "semantic": "语义"}

    hits = {
        item["strategy"] + "/" + item["mode"]: item["overall"]["hit"]
        for item in results
    }

    import numpy as np

    x = np.arange(len(strategies))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = {"keyword": "#d29922", "vector": "#5b8def", "hybrid": "#3fb950"}

    for index, mode in enumerate(modes):
        values = [hits[f"{strategy}/{mode}"] for strategy in strategies]
        offset = (index - 1) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=mode_labels[mode],
            color=colors[mode],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel("Hit@2", fontsize=11)
    ax.set_title("实验二　切分策略 × 检索策略：检索命中率对比", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([strategy_labels[s] for s in strategies], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, frameon=False)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / "exp2_retrieval.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  exp2_retrieval.png")


def chart_exp3() -> None:
    """实验三：静态路径 vs 动态规划。"""
    data = json.loads((EXPERIMENTS_DIR / "exp3_planning.json").read_text(encoding="utf-8"))
    summary = data["summary"]

    plt = _setup_matplotlib()

    labels = [summary[key]["label"] for key in ("static", "dynamic")]
    mean_target = [summary[key]["mean_target"]["mean"] for key in ("static", "dynamic")]
    effective = [summary[key]["effective_ratio"]["mean"] for key in ("static", "dynamic")]

    import numpy as np

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bar1 = ax.bar(x - width / 2, mean_target, width, label="目标知识点均分", color="#5b8def")
    bar2 = ax.bar(x + width / 2, effective, width, label="有效练习占比", color="#3fb950")

    for bars in (bar1, bar2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_ylabel("数值", fontsize=11)
    ax.set_title("实验三　静态路径 vs 动态规划（30 次仿真，30 轮预算）", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, frameon=False)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / "exp3_planning.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  exp3_planning.png")


def main() -> None:
    missing = [
        name
        for name in ("exp1_ablation.json", "exp2_retrieval.json", "exp3_planning.json")
        if not (EXPERIMENTS_DIR / name).exists()
    ]
    if missing:
        print("缺实验结果文件，先跑对应实验：")
        for name in missing:
            print(f"  - {name}")
        print()
        print("运行：")
        print("  .venv\\Scripts\\python.exe experiments\\exp1_memory_ablation.py")
        print("  .venv\\Scripts\\python.exe experiments\\exp2_retrieval.py")
        print("  .venv\\Scripts\\python.exe experiments\\exp3_planning.py")
        sys.exit(1)

    print("生成实验图表：")
    chart_exp1()
    chart_exp2()
    chart_exp3()
    print()
    print(f"全部输出到 {CHARTS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
