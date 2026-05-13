#!/usr/bin/env python
"""Run walk-forward optimization for one stock."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.config import build_bt_kwargs
from src.backtest.wfo import run_nested_walk_forward_optimization, run_walk_forward_optimization
from src.data_fetcher.db_manager import DuckDBManager
from src.indicators import DKTrendParams, TrendMode
from src.settings import load_config, project_root


def _params(cfg: dict, mode: str) -> DKTrendParams:
    raw = dict(cfg.get("trend_signal", {}) or {})
    raw["mode"] = mode
    return DKTrendParams.from_mapping(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DK trend walk-forward optimization.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--oos-days", type=int, default=126)
    parser.add_argument("--mode", choices=[m.value for m in TrendMode], default="macd_cross")
    parser.add_argument("--window", choices=["rolling", "expanding"], default="rolling")
    parser.add_argument("--nested", action="store_true", help="Use nested WFO (outer→inner) for honest OOS evaluation")
    parser.add_argument("--inner-train-days", type=int, default=504, help="Inner WFO train days (for --nested)")
    parser.add_argument("--inner-oos-days", type=int, default=126, help="Inner WFO OOS days (for --nested)")
    parser.add_argument("--export-results", action="store_true")
    parser.add_argument("--plot-heatmap", action="store_true")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    wfo_cfg = cfg.get("wfo", {}) or {}
    symbol = str(args.symbol).strip().zfill(6)
    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        df = db.read_daily_frame(symbols=[symbol], start=args.start, end=args.end)
    if df.empty:
        raise SystemExit("no daily data found; run scripts/fetch_stock.py first")

    bt_kwargs = build_bt_kwargs(cfg)
    # WFO search overrides the base trend mode; clear consensus if not in grid.
    if cfg.get("trend_signal", {}).get("mode") != "consensus":
        bt_kwargs["consensus_n_agree"] = None

    if args.nested:
        result = run_nested_walk_forward_optimization(
            symbol,
            df,
            base_params=_params(cfg, args.mode),
            param_grid=wfo_cfg.get("param_grid"),
            outer_train_days=args.train_days,
            outer_oos_days=args.oos_days,
            inner_train_days=args.inner_train_days,
            inner_oos_days=args.inner_oos_days,
            mode=args.mode,
            window=args.window,
            cost_bps=bt_kwargs.pop("cost_bps"),
            initial_capital=bt_kwargs.pop("initial_capital"),
            bt_kwargs=bt_kwargs,
        )

        agg = result["aggregated"]
        print(f"{symbol} Nested WFO | mode={args.mode} | outer_folds={result['n_outer_folds']}")
        print(
            "OOS combined (truly unseen): "
            f"ann={agg.get('annualized_return', float('nan')):.4f} "
            f"sharpe={agg.get('sharpe_ratio', float('nan')):.2f} "
            f"mdd={agg.get('max_drawdown', float('nan')):.4f}"
        )

        drift = result.get("parameter_drift", {})
        if drift.get("n_folds", 0) >= 2:
            print(
                f"Param drift across outer folds: mean={drift.get('mean_drift', float('nan')):.3f} "
                f"max={drift.get('max_drift', float('nan')):.3f}"
            )

        outer_folds = result.get("outer_folds", [])
        if outer_folds:
            print("\nOuter fold details:")
            for of in outer_folds:
                print(
                    f"  Fold {of['fold']}: {of['outer_oos_start']} → {of['outer_oos_end']} "
                    f"sharpe={of['oos_sharpe']:.2f} ann={of['oos_annualized_return']:.4f} "
                    f"mdd={of['oos_max_drawdown']:.4f} trades={of['oos_n_trades']}"
                )
    else:
        result = run_walk_forward_optimization(
            symbol,
            df,
            base_params=_params(cfg, args.mode),
            param_grid=wfo_cfg.get("param_grid"),
            train_days=args.train_days,
            oos_days=args.oos_days,
            mode=args.mode,
            window=args.window,
            cost_bps=bt_kwargs.pop("cost_bps"),
            initial_capital=bt_kwargs.pop("initial_capital"),
            bt_kwargs=bt_kwargs,
        )

        agg = result["aggregated"]
        print(f"{symbol} WFO | mode={args.mode} | folds={result['n_folds']}")
        print(
            "OOS combined: "
            f"total={agg.get('total_return_combined', float('nan')):.4f} "
            f"ann={agg.get('annualized_return_combined', float('nan')):.4f} "
            f"sharpe={agg.get('sharpe_ratio_combined', float('nan')):.2f} "
            f"mdd={agg.get('max_drawdown_combined', float('nan')):.4f}"
        )

        # Platform stability summary
        drift = result.get("parameter_drift", {})
        if drift.get("n_folds", 0) >= 2:
            print(
                f"Param drift: mean={drift.get('mean_drift', float('nan')):.3f} "
                f"median={drift.get('median_drift', float('nan')):.3f} "
                f"max={drift.get('max_drift', float('nan')):.3f}"
            )
        stability = result.get("parameter_stability", {})
        is_oos_corr = stability.get("is_oos_score_corr", float("nan"))
        print(f"IS/OOS score corr: {is_oos_corr:.3f} {'⚠️ 过拟合风险' if is_oos_corr > 0.7 else '✅'}")

        platform_by_fold = result.get("platform_by_fold", [])
        n_isolated = sum(1 for pf in platform_by_fold if pf.get("platform_info", {}).get("is_isolated"))
        if platform_by_fold:
            print(f"Platform: {n_isolated}/{len(platform_by_fold)} folds picked non-peak platform (isolated peak avoided)")

    if args.export_results:
        out_dir = project_root() / str(cfg.get("paths", {}).get("output_dir", "data/output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{symbol}_wfo_{datetime.now().strftime('%Y%m%d')}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"已写入 {path}")

    if args.plot_heatmap:
        import base64
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        heatmaps = result.get("heatmaps", [])
        param_grid = result.get("param_grid", {})

        if heatmaps:
            # Use pre-computed heatmap data from WFO
            for hm in heatmaps:
                xk, yk = hm["x_param"], hm["y_param"]
                x_vals = hm["x_vals"]
                y_vals = hm["y_vals"]
                is_grid = np.array(hm["z_is_score"])
                oos_grid = np.array(hm.get("z_oos_sharpe", []))

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                for ax, data, title in [
                    (axes[0], is_grid, f"IS Score ({xk} × {yk})"),
                    (axes[1], oos_grid if oos_grid.size else is_grid, f"OOS Sharpe ({xk} × {yk})"),
                ]:
                    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", origin="upper")
                    ax.set_xticks(range(len(x_vals)))
                    ax.set_xticklabels([str(v) for v in x_vals], fontsize=8)
                    ax.set_yticks(range(len(y_vals)))
                    ax.set_yticklabels([str(v) for v in y_vals], fontsize=8)
                    ax.set_xlabel(xk)
                    ax.set_ylabel(yk)
                    ax.set_title(title)
                    for i in range(len(y_vals)):
                        for j in range(len(x_vals)):
                            v = data[i, j]
                            if np.isfinite(v):
                                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                                        color="#222" if abs(v) < 1.5 else "#fff")
                    plt.colorbar(im, ax=ax, shrink=0.8)
                fig.suptitle(f"{symbol} WFO Parameter Heatmap", fontsize=13)
                fig.tight_layout()

                out_dir = project_root() / str(cfg.get("paths", {}).get("output_dir", "data/output"))
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"{symbol}_wfo_heatmap_{xk}_{yk}_{datetime.now().strftime('%Y%m%d')}.html"
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                plt.close(fig)
                path.write_text(
                    f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>WFO Heatmap</title></head>"
                    f"<body><h2>{symbol} WFO 参数热力图 ({xk} × {yk})</h2>"
                    f"<img src='data:image/png;base64,{b64}' style='max-width:100%'></body></html>",
                    encoding="utf-8",
                )
                print(f"热力图已写入 {path}")
        else:
            # Fallback: old-style MACD-only heatmap
            best_by_fold = result.get("best_params_by_fold", [])
            fast_vals = param_grid.get("macd_fast", [])
            slow_vals = param_grid.get("macd_slow", [])
            if fast_vals and slow_vals and best_by_fold:
                is_grid = np.full((len(slow_vals), len(fast_vals)), np.nan)
                oos_grid = np.full((len(slow_vals), len(fast_vals)), np.nan)
                for bf in best_by_fold:
                    p = bf.get("params", {})
                    fi = fast_vals.index(p.get("macd_fast")) if p.get("macd_fast") in fast_vals else -1
                    si = slow_vals.index(p.get("macd_slow")) if p.get("macd_slow") in slow_vals else -1
                    if fi >= 0 and si >= 0:
                        if not np.isfinite(is_grid[si, fi]):
                            is_grid[si, fi] = bf.get("is_score", np.nan)
                            oos_grid[si, fi] = bf.get("oos_sharpe", np.nan)

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                for ax, data, title in [
                    (axes[0], is_grid, "IS Score"),
                    (axes[1], oos_grid, "OOS Sharpe"),
                ]:
                    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", origin="upper")
                    ax.set_xticks(range(len(fast_vals)))
                    ax.set_xticklabels(fast_vals, fontsize=8)
                    ax.set_yticks(range(len(slow_vals)))
                    ax.set_yticklabels(slow_vals, fontsize=8)
                    ax.set_xlabel("macd_fast")
                    ax.set_ylabel("macd_slow")
                    ax.set_title(title)
                    for i in range(len(slow_vals)):
                        for j in range(len(fast_vals)):
                            v = data[i, j]
                            if np.isfinite(v):
                                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                                        color="#222" if abs(v) < 1.5 else "#fff")
                    plt.colorbar(im, ax=ax, shrink=0.8)
                fig.suptitle(f"{symbol} WFO Parameter Heatmap", fontsize=13)
                fig.tight_layout()

                out_dir = project_root() / str(cfg.get("paths", {}).get("output_dir", "data/output"))
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"{symbol}_wfo_heatmap_{datetime.now().strftime('%Y%m%d')}.html"
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                plt.close(fig)
                path.write_text(
                    f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>WFO Heatmap</title></head>"
                    f"<body><h2>{symbol} WFO 参数热力图</h2>"
                    f"<img src='data:image/png;base64,{b64}' style='max-width:100%'></body></html>",
                    encoding="utf-8",
                )
                print(f"热力图已写入 {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
