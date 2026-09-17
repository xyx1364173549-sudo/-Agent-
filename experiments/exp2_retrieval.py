"""实验二：切分策略 × 检索策略 的交叉对比。

对应论文第 6 章实验二，也是 M3 留下的那个伏笔的正式量化。

## 实验设计

**自变量**两个：

    切分策略   固定长度 / 递归 / 语义 —— 决定「块怎么切」
    检索策略   关键词 / 向量 / 混合 —— 决定「块怎么找」

**因变量**是检索质量，用三个标准指标衡量（见 ``src/eval/metrics.py``）：

    Hit@2      前 2 条里有没有命中相关文档
    MRR        第一条命中的名次倒数
    Recall@2   前 2 条覆盖了相关文档的几成

**按查询难度分档统计**，这是本实验的关键设计。查询与目标文档的字面重叠度
由 ``literal_overlap`` 客观算出，分高 / 中 / 低三档：

    高重叠    几乎照抄资料的说法 —— 关键词检索的主场
    低重叠    用生活化的说法问同一件事 —— 考验「理解意思」的能力

## 两个被实测纠正的设计

**top_k 取 3 时各策略分数全是 1.0**。8 篇资料里取 3 条，随手就蒙中了，
差距被天花板效应抹平。改成取 2 条才有区分度。

**原本想按「字面型 / 语义型」二分查询，分不开**。实测发现中文技术资料的
领域词太集中，只要问的是同一件事，几乎必然撞上几个字（「运算量随数据量
怎么涨」的重叠度高达 0.78）。强行二分只会得到一张全在一类的表格。
改成分档统计连续的重叠度，信息量更大。

## 零 API 调用

检索本身（向量化 + 相似度计算）全在本地跑，不经过大模型。
所以这个实验可以随便重跑，结果也完全可复现。

运行::

    .venv\\Scripts\\python.exe experiments\\exp2_retrieval.py
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

from scripts.build_kb import build  # noqa: E402
from src.eval import (  # noqa: E402
    MATERIALS,
    RETRIEVAL_QUERIES,
    hit_at_k,
    literal_overlap,
    mean,
    recall_at_k,
    reciprocal_rank,
)
from src.rag import Retriever, VectorStore  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 自变量取值
STRATEGIES = ("fixed", "recursive", "semantic")
MODES = ("keyword", "vector", "hybrid")

# 切分参数。资料每篇两三百字，块切得小一些才看得清切分策略的差别
CHUNK_SIZE = 150
OVERLAP = 30

# 只取前 2 条。取 3 条的话，8 篇资料里随手就能蒙中，
# 各策略的差距会被天花板效应抹平——第一版就是这么栽的
TOP_K = 2

# 重叠度分档。分界线取在实测的两个空档上：「几乎照抄」的查询重叠度
# 在 0.45 以上，「换了说法」的落在 0.25 以下
HIGH_THRESHOLD = 0.45
MID_THRESHOLD = 0.25

GRADES = ("high", "mid", "low")
GRADE_LABEL = {"high": "高重叠", "mid": "中重叠", "low": "低重叠"}

OUTPUT_DIR = PROJECT_ROOT / "docs" / "experiments"

MODE_LABEL = {"keyword": "关键词", "vector": "向量", "hybrid": "混合"}
STRATEGY_LABEL = {"fixed": "固定长度", "recursive": "递归", "semantic": "语义"}


def grade_of(overlap: float) -> str:
    """按重叠度分档。"""
    if overlap >= HIGH_THRESHOLD:
        return "high"
    if overlap >= MID_THRESHOLD:
        return "mid"
    return "low"


def overlap_of(query: str, relevant: list[str]) -> float:
    """查询与目标文档的最大字面重叠度。

    不靠人拍脑袋判断「这个查询难不难」——写查询的人自己知道答案，
    很容易觉得「这个肯定算难」，实际却和文档重合了好几个字。
    """
    return max(literal_overlap(query, MATERIALS[name]) for name in relevant)


def write_materials(target: Path) -> Path:
    """把数据集里的资料写到临时目录，供构建脚本读取。"""
    target.mkdir(parents=True, exist_ok=True)
    for name, content in MATERIALS.items():
        (target / name).write_text(content, encoding="utf-8")
    return target


def evaluate(strategy: str, mode: str, workdir: Path) -> dict:
    """跑一个实验条件：指定切分策略 + 指定检索模式，返回各项指标。"""
    materials = write_materials(workdir / "materials")

    # 每个条件用独立的向量库目录，避免上一个条件的块混进来
    store = VectorStore(
        persist_dir=workdir / f"chroma-{strategy}",
        collection_name="exp_kb",
    )
    build(
        materials,
        strategy=strategy,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        store=store,
        reset=True,
    )

    retriever = Retriever(store)
    rows: list[dict] = []

    for item in RETRIEVAL_QUERIES:
        hits = retriever.search(item["query"], mode=mode, top_k=TOP_K)
        sources = [hit.get("source", "") for hit in hits]
        overlap = overlap_of(item["query"], item["relevant"])

        rows.append(
            {
                "query": item["query"],
                "grade": grade_of(overlap),
                "overlap": overlap,
                "expected": item["relevant"],
                "returned": sources,
                "hit": hit_at_k(sources, item["relevant"], TOP_K),
                "rr": reciprocal_rank(sources, item["relevant"]),
                "recall": recall_at_k(sources, item["relevant"], TOP_K),
            }
        )

    return {
        "strategy": strategy,
        "mode": mode,
        "chunks": store.count(),
        "overall": _aggregate(rows),
        "by_grade": {
            grade: _aggregate([row for row in rows if row["grade"] == grade])
            for grade in GRADES
        },
        "rows": rows,
    }


def _aggregate(rows: list[dict]) -> dict:
    """把若干查询的指标求平均。"""
    if not rows:
        return {"hit": 0.0, "mrr": 0.0, "recall": 0.0, "n": 0}

    return {
        "hit": mean([row["hit"] for row in rows]),
        "mrr": mean([row["rr"] for row in rows]),
        "recall": mean([row["recall"] for row in rows]),
        "n": len(rows),
    }


def print_table(results: list[dict]) -> None:
    """打印结果表。列宽按最长的标签对齐，看着舒服一点。"""
    print()
    print("=" * 88)
    print(f"实验二　切分策略 × 检索策略　（Hit@{TOP_K} / MRR / Recall@{TOP_K}）")
    print("=" * 88)
    print(f"切分参数：chunk_size={CHUNK_SIZE}　overlap={OVERLAP}　top_k={TOP_K}")
    print()

    header = (
        f"{'切分策略':<10}{'检索':<8}{'块数':>5}   "
        f"{'Hit@2':>7}{'MRR':>8}{'Recall':>8}   "
        f"{'高重叠':>7}{'中重叠':>7}{'低重叠':>7}"
    )
    print(header)
    print("-" * 92)

    for item in results:
        overall = item["overall"]
        print(
            f"{STRATEGY_LABEL[item['strategy']]:<10}"
            f"{MODE_LABEL[item['mode']]:<8}"
            f"{item['chunks']:>5}   "
            f"{overall['hit']:>7.3f}{overall['mrr']:>8.3f}{overall['recall']:>8.3f}   "
            f"{item['by_grade']['high']['hit']:>7.3f}"
            f"{item['by_grade']['mid']['hit']:>7.3f}"
            f"{item['by_grade']['low']['hit']:>7.3f}"
        )

    print()
    print("逐条明细（重叠度 = 查询的二元组在目标文档中出现的比例）：")

    for row in results[0]["rows"]:
        mark = "✓" if row["hit"] else "✗"
        print(
            f"  {mark} [{GRADE_LABEL[row['grade']]} {row['overlap']:.2f}] {row['query']}"
        )
        if not row["hit"]:
            print(f"      期望 {row['expected']}　实际返回 {row['returned'] or '（空）'}")
    print()

    # 挑出最有说服力的对比，直接给结论，省得读者自己去表里找
    print("关键对比（向量 − 关键词）：")
    for strategy in STRATEGIES:
        by_mode = {item["mode"]: item for item in results if item["strategy"] == strategy}
        if "keyword" not in by_mode or "vector" not in by_mode:
            continue
        low_gap = by_mode["vector"]["by_grade"]["low"]["hit"] - by_mode["keyword"]["by_grade"]["low"]["hit"]
        mrr_gap = by_mode["vector"]["overall"]["mrr"] - by_mode["keyword"]["overall"]["mrr"]
        print(
            f"  {STRATEGY_LABEL[strategy]:<6}　"
            f"低重叠查询 {low_gap:+.3f}　整体 MRR {mrr_gap:+.3f}"
        )
    print()

    print("各档样本数（三种检索模式共用同一批查询）：")
    for grade in GRADES:
        count = results[0]["by_grade"][grade]["n"]
        print(f"  {GRADE_LABEL[grade]}：{count} 条")
    print()


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="exp2-"))
    results: list[dict] = []

    try:
        for strategy in STRATEGIES:
            for mode in MODES:
                print(f"  跑 {STRATEGY_LABEL[strategy]} × {MODE_LABEL[mode]} ……", flush=True)
                results.append(evaluate(strategy, mode, workdir))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print_table(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "exp2_retrieval.json"
    out_file.write_text(
        json.dumps(
            {
                "config": {"chunk_size": CHUNK_SIZE, "overlap": OVERLAP, "top_k": TOP_K},
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"结果已写入 {out_file.relative_to(PROJECT_ROOT)}")
    print("（这个实验不调用大模型，可以随便重跑）")


if __name__ == "__main__":
    main()
