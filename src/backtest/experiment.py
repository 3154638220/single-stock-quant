"""Experiment management: directory scaffold, index, and decision rules."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


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
    r = rules or DEFAULT_DECISION_RULES
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
