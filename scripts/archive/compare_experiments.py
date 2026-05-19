#!/usr/bin/env python
"""Compare two experiment directories and export a delta report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.experiment import (
    compare_metric_summaries,
    load_experiment_metrics,
    render_experiment_comparison_html,
    write_delta_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two single-stock-quant experiment directories.")
    parser.add_argument("--baseline", required=True, help="Baseline experiment directory")
    parser.add_argument("--current", required=True, help="Current experiment directory")
    parser.add_argument("--output", required=True, help="Output report path (.html or .csv)")
    parser.add_argument("--export-csv", help="Optional CSV path for the comparison table")
    parser.add_argument("--write-delta", action="store_true", help="Write DELTA.md into the current experiment directory")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline).expanduser()
    current_dir = Path(args.current).expanduser()
    output = Path(args.output).expanduser()

    if not baseline_dir.exists():
        raise SystemExit(f"baseline experiment directory does not exist: {baseline_dir}")
    if not current_dir.exists():
        raise SystemExit(f"current experiment directory does not exist: {current_dir}")

    baseline_metrics = load_experiment_metrics(baseline_dir)
    current_metrics = load_experiment_metrics(current_dir)
    if not baseline_metrics and not current_metrics:
        raise SystemExit("no comparable metrics found in either experiment directory")

    rows = compare_metric_summaries(baseline_metrics, current_metrics)
    table = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() == ".csv":
        table.to_csv(output, index=False)
    else:
        html = render_experiment_comparison_html(
            rows,
            baseline_dir=baseline_dir,
            current_dir=current_dir,
        )
        output.write_text(html, encoding="utf-8")
        csv_path = Path(args.export_csv).expanduser() if args.export_csv else output.with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_path, index=False)

    if args.write_delta:
        write_delta_markdown(
            rows,
            baseline_dir=baseline_dir,
            current_dir=current_dir,
            output_path=current_dir / "DELTA.md",
        )

    improved = int((table["direction"] == "improved").sum()) if not table.empty else 0
    worse = int((table["direction"] == "worse").sum()) if not table.empty else 0
    print(f"Compared {len(table)} metrics: improved={improved}, worse={worse}")
    print(f"Report written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
