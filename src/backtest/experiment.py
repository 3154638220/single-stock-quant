"""Experiment management: directory scaffold, index, and decision rules."""

from __future__ import annotations

import hashlib
import html
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_EXPERIMENT_ARTIFACTS = (
    "config.yaml",
    "batch_summary.csv",
    "wfo_summary.csv",
    "trade_attribution.csv",
    "meta_label_calibration.csv",
    "feature_importance.csv",
    "regime_breakdown.csv",
    "stability_heatmap.html",
    "report.html",
    "DELTA.md",
)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def config_hash(cfg: dict[str, Any]) -> str:
    raw = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def expected_experiment_artifacts() -> list[str]:
    """Return the standard experiment files tracked by the research loop."""
    return list(EXPECTED_EXPERIMENT_ARTIFACTS)


def write_experiment_manifest(exp_dir: Path | str) -> Path:
    """Write a lightweight manifest of expected experiment artifacts."""
    path = Path(exp_dir) / "ARTIFACTS.md"
    lines = ["# Experiment Artifacts", ""]
    for artifact in EXPECTED_EXPERIMENT_ARTIFACTS:
        lines.append(f"- `{artifact}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def create_experiment_dir(
    experiment_id: str,
    *,
    base_dir: Path | str | None = None,
    cfg: dict[str, Any] | None = None,
    notes: str = "",
) -> Path:
    """Create a standard experiment output directory.

    Returns the Path to the experiment directory.
    """
    if base_dir is None:
        base_dir = Path("data/output/experiments")
    base = Path(base_dir)
    date_prefix = datetime.now().strftime("%Y%m%d")
    name = f"{date_prefix}_{experiment_id}"
    exp_dir = base / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Write config snapshot
    if cfg is not None:
        import yaml
        cfg_path = exp_dir / "config.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(dict(cfg), f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # Write notes
    notes_path = exp_dir / "notes.md"
    notes_path.write_text(
        f"# {name}\n\n{notes}\n",
        encoding="utf-8",
    )
    write_experiment_manifest(exp_dir)
    delta_path = exp_dir / "DELTA.md"
    if not delta_path.exists():
        delta_path.write_text(
            "# Delta\n\nNo baseline comparison has been generated yet.\n",
            encoding="utf-8",
        )

    return exp_dir


def update_experiment_index(
    experiment_id: str,
    *,
    exp_dir: Path,
    cfg: dict[str, Any] | None = None,
    start: str = "",
    end: str = "",
    universe: str = "",
    metrics: dict[str, float] | None = None,
    notes: str = "",
    index_path: Path | str | None = None,
) -> Path:
    """Add or update an entry in the experiment index CSV."""
    if index_path is None:
        index_path = exp_dir.parent / "index.csv"
    idx = Path(index_path)

    commit = get_git_commit()
    ch = config_hash(cfg) if cfg else ""

    row = {
        "experiment_id": experiment_id,
        "git_commit": commit,
        "config_hash": ch,
        "start": str(start),
        "end": str(end),
        "universe": str(universe),
        "timestamp": datetime.now().isoformat(),
        "notes": notes,
    }
    if metrics:
        row.update(metrics)

    new = pd.DataFrame([row])
    if idx.exists():
        existing = pd.read_csv(idx)
        existing = existing[existing["experiment_id"] != experiment_id]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new

    combined.to_csv(idx, index=False)
    return idx


DEFAULT_DECISION_RULES = {
    "hypothesis": "",
    "pass_conditions": [],
    "fail_conditions": [
        "全样本收益提升但OOS恶化 → 放弃",
        "Sharpe提升但最大回撤恶化>5pct → 只保留为高风险变体",
        "交易次数下降到原来的20%以下 → 视为样本不足",
        "只在单一股票有效 → 不作为通用策略",
    ],
}


def evaluate_experiment(
    metrics: dict[str, float],
    baseline: dict[str, float] | None = None,
    *,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply decision rules to an experiment's metrics.

    Returns a dict with ``decision``, ``warnings``, and ``verdict``.
    """
    warnings = []
    verdict = "pass"

    if baseline:
        sharpe_change = metrics.get("median_sharpe", 0.0) - baseline.get("median_sharpe", 0.0)
        mdd_change = metrics.get("median_max_drawdown", 0.0) - baseline.get("median_max_drawdown", 0.0)
        trades_ratio = metrics.get("median_n_trades", 1.0) / max(baseline.get("median_n_trades", 1.0), 1)

        if sharpe_change > 0 and mdd_change > 0.05:
            warnings.append("Sharpe提升但最大回撤恶化了>5pct")
            verdict = "high_risk_variant_only"
        if trades_ratio < 0.20:
            warnings.append(f"交易次数降至baseline的{trades_ratio:.1%}")
            verdict = "insufficient_samples"
        if sharpe_change < 0:
            warnings.append("Sharpe未改善")

    return {
        "decision": "accept" if verdict == "pass" else f"flag: {verdict}",
        "warnings": warnings,
        "verdict": verdict,
    }


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"symbol", "code", "id", "fold"}
    cols = []
    for col in df.select_dtypes(include="number").columns:
        name = str(col)
        if name.lower() not in excluded:
            cols.append(name)
    return cols


def _add_median_summary(metrics: dict[str, float], prefix: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    metrics[f"{prefix}_n_rows"] = float(len(df))
    for col in _numeric_columns(df):
        metrics[f"{prefix}_median_{col}"] = _safe_float(df[col].median())


def load_experiment_metrics(exp_dir: Path | str) -> dict[str, float]:
    """Load comparable numeric metrics from a standard experiment directory.

    The loader is intentionally tolerant: missing files are ignored, and
    batch/WFO summaries are aggregated by median across symbols or folds.
    """
    root = Path(exp_dir)
    metrics: dict[str, float] = {}

    batch_path = root / "batch_summary.csv"
    if batch_path.exists():
        batch = pd.read_csv(batch_path)
        if "status" in batch.columns:
            batch = batch[batch["status"].astype(str).str.lower().eq("ok")]
        _add_median_summary(metrics, "batch", batch)

    wfo_path = root / "wfo_summary.csv"
    if wfo_path.exists():
        _add_median_summary(metrics, "wfo", pd.read_csv(wfo_path))

    calibration_path = root / "meta_label_calibration.csv"
    if calibration_path.exists():
        _add_median_summary(metrics, "meta_label", pd.read_csv(calibration_path))

    feature_path = root / "feature_importance.csv"
    if feature_path.exists():
        feature_df = pd.read_csv(feature_path)
        if {"feature", "importance"}.issubset(feature_df.columns):
            top = feature_df.sort_values("importance", ascending=False).head(5)
            metrics["feature_importance_top5_sum"] = _safe_float(top["importance"].sum())
        _add_median_summary(metrics, "feature_importance", feature_df)

    regime_path = root / "regime_breakdown.csv"
    if regime_path.exists():
        _add_median_summary(metrics, "regime", pd.read_csv(regime_path))

    return metrics


def _higher_is_better(metric: str) -> bool:
    lower_is_better_tokens = (
        "drawdown",
        "mdd",
        "turnover",
        "cost",
        "loss",
        "volatility",
    )
    name = metric.lower()
    return not any(token in name for token in lower_is_better_tokens)


def _ci_overlap(
    metric: str,
    baseline: dict[str, float],
    current: dict[str, float],
) -> str:
    b_lo = baseline.get(f"{metric}_ci_low")
    b_hi = baseline.get(f"{metric}_ci_high")
    c_lo = current.get(f"{metric}_ci_low")
    c_hi = current.get(f"{metric}_ci_high")
    vals = [b_lo, b_hi, c_lo, c_hi]
    if any(v is None or not math.isfinite(float(v)) for v in vals):
        return ""
    return "yes" if max(float(b_lo), float(c_lo)) <= min(float(b_hi), float(c_hi)) else "no"


def compare_metric_summaries(
    baseline: dict[str, float],
    current: dict[str, float],
) -> list[dict[str, Any]]:
    """Build row-wise metric deltas between two experiment summaries."""
    rows: list[dict[str, Any]] = []
    keys = sorted(set(baseline) | set(current))
    for key in keys:
        if key.endswith("_ci_low") or key.endswith("_ci_high"):
            continue
        base = _safe_float(baseline.get(key))
        curr = _safe_float(current.get(key))
        if not math.isfinite(base) and not math.isfinite(curr):
            continue
        delta = curr - base if math.isfinite(base) and math.isfinite(curr) else float("nan")
        delta_pct = (
            delta / abs(base)
            if math.isfinite(delta) and math.isfinite(base) and abs(base) > 1e-12
            else float("nan")
        )
        higher_better = _higher_is_better(key)
        if not math.isfinite(delta) or abs(delta) < 1e-12:
            direction = "flat"
        elif (delta > 0 and higher_better) or (delta < 0 and not higher_better):
            direction = "improved"
        else:
            direction = "worse"
        rows.append(
            {
                "metric": key,
                "baseline": base,
                "current": curr,
                "delta": delta,
                "delta_pct": delta_pct,
                "direction": direction,
                "ci_overlap": _ci_overlap(key, baseline, current),
            }
        )
    return rows


def write_delta_markdown(
    rows: list[dict[str, Any]],
    *,
    baseline_dir: Path | str,
    current_dir: Path | str,
    output_path: Path | str,
) -> Path:
    """Write a compact DELTA.md summary for the current experiment."""
    out = Path(output_path)
    improved = sum(1 for row in rows if row.get("direction") == "improved")
    worse = sum(1 for row in rows if row.get("direction") == "worse")
    flat = sum(1 for row in rows if row.get("direction") == "flat")
    lines = [
        "# Delta",
        "",
        f"- Baseline: `{Path(baseline_dir)}`",
        f"- Current: `{Path(current_dir)}`",
        f"- Improved metrics: {improved}",
        f"- Worse metrics: {worse}",
        f"- Flat metrics: {flat}",
        "",
        "| Metric | Baseline | Current | Delta | Delta % | Direction | CI overlap |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {metric} | {baseline:.6g} | {current:.6g} | {delta:.6g} | {delta_pct:.2%} | {direction} | {ci_overlap} |".format(
                metric=row["metric"],
                baseline=row["baseline"],
                current=row["current"],
                delta=row["delta"],
                delta_pct=row["delta_pct"],
                direction=row["direction"],
                ci_overlap=row["ci_overlap"] or "",
            )
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def render_experiment_comparison_html(
    rows: list[dict[str, Any]],
    *,
    baseline_dir: Path | str,
    current_dir: Path | str,
) -> str:
    """Render a self-contained HTML comparison report."""
    improved = sum(1 for row in rows if row.get("direction") == "improved")
    worse = sum(1 for row in rows if row.get("direction") == "worse")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    def fmt(value: float) -> str:
        return "" if not math.isfinite(value) else f"{value:.6g}"

    table_rows = []
    for row in rows:
        cls = html.escape(str(row["direction"]))
        table_rows.append(
            "<tr class='{cls}'><td>{metric}</td><td>{baseline}</td><td>{current}</td>"
            "<td>{delta}</td><td>{delta_pct}</td><td>{direction}</td><td>{ci_overlap}</td></tr>".format(
                cls=cls,
                metric=html.escape(str(row["metric"])),
                baseline=fmt(float(row["baseline"])),
                current=fmt(float(row["current"])),
                delta=fmt(float(row["delta"])),
                delta_pct="" if not math.isfinite(float(row["delta_pct"])) else f"{float(row['delta_pct']):.2%}",
                direction=html.escape(str(row["direction"])),
                ci_overlap=html.escape(str(row["ci_overlap"] or "")),
            )
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Experiment Comparison</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 1180px; margin: 0 auto; padding: 24px; color: #222; }}
h1 {{ font-size: 22px; margin-bottom: 4px; }}
.subtitle {{ color: #666; font-size: 13px; margin-bottom: 18px; }}
.kpi {{ display: inline-block; min-width: 120px; background: #f7f7f7; border-radius: 6px; padding: 10px 14px; margin-right: 8px; }}
.kpi .v {{ font-size: 20px; font-weight: 700; }}
.kpi .l {{ color: #777; font-size: 11px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 18px; }}
th, td {{ border: 1px solid #e0e0e0; padding: 7px 10px; text-align: right; }}
th {{ background: #f5f5f5; }}
td:first-child, th:first-child {{ text-align: left; }}
tr.improved td {{ background: #f2fbf5; }}
tr.worse td {{ background: #fff5f5; }}
</style>
</head>
<body>
<h1>Experiment Comparison</h1>
<div class="subtitle">Baseline: {html.escape(str(Path(baseline_dir)))}<br>Current: {html.escape(str(Path(current_dir)))}<br>Generated: {generated}</div>
<div>
<div class="kpi"><div class="v">{len(rows)}</div><div class="l">Metrics</div></div>
<div class="kpi"><div class="v">{improved}</div><div class="l">Improved</div></div>
<div class="kpi"><div class="v">{worse}</div><div class="l">Worse</div></div>
</div>
<table>
<tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Delta</th><th>Delta %</th><th>Direction</th><th>CI overlap</th></tr>
{''.join(table_rows)}
</table>
</body>
</html>"""
