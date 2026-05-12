#!/usr/bin/env python3
"""运行前环境自检：Python 版本、核心依赖、DuckDB 可写、AkShare 连通。

用法:
    python scripts/env_check.py
    python scripts/env_check.py --quiet   # 仅退出码，无输出
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_check import run_checks


def main() -> int:
    p = argparse.ArgumentParser(description="单股多空趋势系统环境自检")
    p.add_argument("--config", type=Path, default=None, help="config.yaml 路径")
    p.add_argument("--quiet", action="store_true", help="仅通过退出码表示结果（0=全部通过）")
    args = p.parse_args()
    return run_checks(config=args.config, quiet=args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
