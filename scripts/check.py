"""一键校验脚本。

按 ``AGENTS.md`` 的约定，每次改动交付前都要「测试全绿」。
本脚本把「跑测试 + 重新生成进度看板」串成一条命令，避免漏步骤：

    python scripts/check.py

返回码：测试通过为 0，测试失败为 1（可直接用于 CI 或 git hook）。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Windows 控制台默认 GBK，输出中文可能报 UnicodeEncodeError，这里统一成 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _run(args: list[str], *, title: str) -> int:
    print(f"\n=== {title} ===")
    print("$ " + " ".join(args))
    completed = subprocess.run(args, cwd=PROJECT_ROOT)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    extra = list(argv or [])
    pytest_code = _run([sys.executable, "-m", "pytest", *extra], title="运行测试")

    if pytest_code != 0:
        print("\n[FAIL] 测试未通过，已跳过看板生成。先修测试，再看进度。")
        return 1

    progress_code = _run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "gen_progress.py"), "--print"],
        title="生成进度看板",
    )

    if progress_code != 0:
        print("\n[FAIL] 看板生成失败。")
        return 1

    print("\n[OK] 测试全绿，进度看板已刷新。下一步：git commit。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
