"""遗忘与巩固：让记忆像人一样——久不用的慢慢淡忘，常提到的越记越牢。

一条记忆的「当前强度」由两件事决定::

    强度 = 重要性 × 时间衰减

- **重要性**：写进来时就定了（0~1），越重要越抗忘；
- **时间衰减**：用「半衰期」描述——每过 ``half_life_days`` 天，强度减半。
  这个思路借鉴自艾宾浩斯遗忘曲线：遗忘不是线性的，而是先快后慢。

举例（半衰期 7 天）::

    一条 importance = 1.0 的事件：当天 1.00 → 7 天后 0.50 → 14 天后 0.25
    一条 importance = 0.4 的事件：当天 0.40 → 7 天后 0.20 → 14 天后 0.10

假设归档阈值是 0.2，那么重要的事能撑十几天，琐碎的事一周就该被忘了。
**这就是我们想要的效果：重要的事记得久，琐碎的事忘得快。**

之所以把「计算」单独放一个文件、而把「改数据」留在 ``episodic.py``：
计算是纯函数，输入相同结果就相同，测起来干净；数据库操作则要处理事务和状态。
"""

from __future__ import annotations

from datetime import datetime

# 默认半衰期（天）。7 天意味着「一周不碰，印象减半」。
DEFAULT_HALF_LIFE_DAYS = 7.0

# 默认归档阈值。强度低于它就认为「已经想不起来了」。
DEFAULT_FORGET_THRESHOLD = 0.2

SECONDS_PER_DAY = 86400.0


def days_since(created_at: str, now: datetime | None = None) -> float:
    """从 ``created_at`` 到现在过了多少天（可以带小数）。

    ``created_at`` 是 ``store.now_iso()`` 写进去的 ISO8601 文本。
    未来时间会返回 0——时钟不准时不该算出负的年龄，那会让强度反而变大。
    """
    moment = now or datetime.now()
    delta = moment - datetime.fromisoformat(created_at)
    return max(0.0, delta.total_seconds() / SECONDS_PER_DAY)


def strength(
    importance: float,
    created_at: str,
    *,
    now: datetime | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """算一条记忆当前的强度，取值 ``0.0 ~ importance``。

    参数
    ----
    importance:
        写入时定的重要程度，``0.0 ~ 1.0``。
    created_at:
        事件创建时间，ISO8601 文本。
    now:
        用来算衰减的「当前时间」。测试时显式传它，结果就完全可控——
        不用等到真的过了一周才能验证「一周后强度减半」。
    half_life_days:
        半衰期天数，必须为正数。越小忘得越快。
    """
    if half_life_days <= 0:
        raise ValueError(f"half_life_days 需为正数，实际为 {half_life_days}")

    age_days = days_since(created_at, now)

    # 0.5 ** (过了几个半衰期)：过一个半衰期剩一半，过两个剩四分之一
    decay = 0.5 ** (age_days / half_life_days)
    return importance * decay
