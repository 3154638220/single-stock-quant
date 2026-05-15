#!/usr/bin/env python
"""Plot DK trend bars from local DuckDB daily data.

Supports single-mode plotting (existing behaviour) and multi-mode comparison
plots activated via ``--compare-modes``.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_fetcher.db_manager import DuckDBManager
from src.data_fetcher.stock_name_cache import resolve_stock_name_cache_path, resolve_stock_names
from src.indicators import DKTrendParams, TrendMode, compute_dktrend
from src.settings import load_config, project_root


def _params(cfg: dict, mode: str | None) -> DKTrendParams:
    raw = dict(cfg.get("trend_signal", {}) or {})
    if mode:
        raw["mode"] = mode
    return DKTrendParams.from_mapping(raw)


def _params_from_spec(spec: str) -> tuple[str, DKTrendParams]:
    """Parse a mode specification string into a (label, params) pair.

    Supported formats::

        long_ma_trend:250          # mode with trend_ma_period
        long_ma_trend:250,slope=5  # mode with period and extra params
        dual_ma_cross:30-120       # mode with fast-slow
        macd_cross                 # mode with defaults
    """
    extra = {}
    if ":" in spec:
        mode_str, rest = spec.split(":", 1)
        parts = rest.split(",")
        for part in parts:
            if "-" in part and not part.startswith("-"):
                nums = part.split("-")
                if len(nums) == 2 and all(n.isdigit() for n in nums):
                    if "dual_ma" in mode_str or "dual" in mode_str:
                        extra["dual_ma_fast"] = int(nums[0])
                        extra["dual_ma_slow"] = int(nums[1])
                    else:
                        extra["trend_ma_period"] = int(nums[0])
                        extra["slope_lookback"] = int(nums[1])
            elif "=" in part:
                k, v = part.split("=", 1)
                extra[k.strip()] = float(v) if "." in v else int(v)
            elif part.isdigit():
                extra["trend_ma_period"] = int(part)
    else:
        mode_str = spec

    if "-" in spec and ":" not in spec:
        raise SystemExit(f"invalid compare spec {spec!r}: expected mode:param or mode:fast-slow")

    mode = TrendMode(mode_str.strip())
    label_parts = [mode.value]
    if "trend_ma_period" in extra:
        label_parts.append(str(extra["trend_ma_period"]))
    if "dual_ma_fast" in extra and "dual_ma_slow" in extra:
        label_parts.append(f"{extra['dual_ma_fast']}-{extra['dual_ma_slow']}")
    label = ":".join(label_parts)

    base = DKTrendParams(mode=mode)
    params = DKTrendParams(
        mode=mode,
        macd_fast=base.macd_fast,
        macd_slow=base.macd_slow,
        macd_signal=base.macd_signal,
        ma_fast=base.ma_fast,
        ma_slow=base.ma_slow,
        ma_smooth=base.ma_smooth,
        boll_window=base.boll_window,
        min_run_len=base.min_run_len,
        donchian_entry_window=base.donchian_entry_window,
        donchian_exit_window=base.donchian_exit_window,
        trend_ma_period=int(extra.get("trend_ma_period", base.trend_ma_period)),
        trend_ma_type=str(extra.get("trend_ma_type", base.trend_ma_type)),
        slope_lookback=int(extra.get("slope_lookback", base.slope_lookback)),
        require_positive_slope=bool(extra.get("require_positive_slope", base.require_positive_slope)),
        dual_ma_fast=int(extra.get("dual_ma_fast", base.dual_ma_fast)),
        dual_ma_slow=int(extra.get("dual_ma_slow", base.dual_ma_slow)),
        trend_score_ma_long=int(extra.get("trend_score_ma_long", base.trend_score_ma_long)),
        trend_score_ma_fast=int(extra.get("trend_score_ma_fast", base.trend_score_ma_fast)),
        trend_score_ma_slow=int(extra.get("trend_score_ma_slow", base.trend_score_ma_slow)),
        min_breakout_days=int(extra.get("min_breakout_days", base.min_breakout_days)),
    )
    return label, params


def _default_output_path(cfg: dict, symbol: str, suffix: str) -> Path:
    out_dir = project_root() / str(cfg.get("paths", {}).get("output_dir", "data/output"))
    return out_dir / f"{symbol}_{suffix}_dktrend.png"


def _resolve_output_path(raw: str | None, cfg: dict, symbol: str, suffix: str) -> Path:
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = project_root() / path
        return path
    return _default_output_path(cfg, symbol, suffix)


def _load_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "single-stock-quant-mpl"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.font_manager as font_manager
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with: pip install matplotlib>=3.8"
        ) from exc

    has_cjk = _configure_cjk_font(plt, font_manager)
    return mdates, plt, has_cjk


def _configure_cjk_font(plt, font_manager) -> bool:
    names = {font.name for font in font_manager.fontManager.ttflist}
    for family in [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]:
        if family in names:
            plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def _ascii_fallback(text: str) -> str:
    if text.isascii():
        return text
    out = "".join(ch for ch in text if ch.isascii()).strip()
    return out or "DK Trend"


def _plot(trend: pd.DataFrame, *, title: str, mode: str, output_path: Path) -> None:
    mdates, plt, has_cjk = _load_matplotlib()
    plot_df = trend.copy()
    plot_df["trade_date"] = pd.to_datetime(plot_df["trade_date"])
    colors = plot_df["dk_color"].map({"red": "#d62728", "green": "#2ca02c"}).fillna("#808080")

    fig, (ax_price, ax_trend) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    ax_price.plot(plot_df["trade_date"], plot_df["close"], color="#1f2937", linewidth=1.4)
    display_title = title if has_cjk else _ascii_fallback(title)
    ax_price.set_title(f"{display_title} DK Trend ({mode})")
    ax_price.set_ylabel("Close")
    ax_price.grid(True, alpha=0.25)

    ax_trend.bar(
        plot_df["trade_date"],
        plot_df["dk_value"],
        color=colors,
        width=0.8,
        align="center",
    )
    ax_trend.axhline(0.0, color="#4b5563", linewidth=0.8)
    ax_trend.set_ylabel("DK")
    ax_trend.grid(True, axis="y", alpha=0.25)
    ax_trend.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_comparison(df: pd.DataFrame, results: list[tuple[str, pd.DataFrame]], *,
                     title: str, output_path: Path) -> None:
    """Plot multiple trend-line candidates on shared axes for visual comparison."""
    mdates, plt, has_cjk = _load_matplotlib()

    plot_df = df.copy()
    plot_df["trade_date"] = pd.to_datetime(plot_df["trade_date"])

    n = len(results)
    fig = plt.figure(figsize=(16, 4 + 2.5 * n), constrained_layout=True)
    gs = fig.add_gridspec(n + 2, 1, height_ratios=[2.5] + [1] * n + [1], hspace=0.35)

    ax_price = fig.add_subplot(gs[0])
    ax_price.plot(plot_df["trade_date"], plot_df["close"],
                  color="#1f2937", linewidth=1.0, label="Close")
    display_title = title if has_cjk else _ascii_fallback(title)
    ax_price.set_title(f"{display_title} 多空趋势对比")
    ax_price.set_ylabel("Close")
    ax_price.grid(True, alpha=0.25)

    colors = ["#FF6B00", "#0066CC", "#00AA44", "#AA00CC", "#CC6600"]
    color_bars = ["#FF6B00", "#0066CC", "#00AA44", "#AA00CC", "#CC6600"]

    for i, (label, trend) in enumerate(results):
        tdf = trend.copy()
        tdf["trade_date"] = pd.to_datetime(tdf["trade_date"])
        c = colors[i % len(colors)]

        ax_t = fig.add_subplot(gs[i + 1], sharex=ax_price)
        bar_colors = tdf["dk_color"].map({"red": c, "green": "#aaaaaa"}).fillna("#cccccc")
        ax_t.bar(tdf["trade_date"], tdf["dk_value"], color=bar_colors,
                 width=0.8, align="center")
        ax_t.axhline(0.0, color="#4b5563", linewidth=0.6, alpha=0.6)
        n_signals = (tdf["dk_signal"] != "").sum()
        switches = (tdf["dk_color"] != tdf["dk_color"].shift(1)).sum()
        total_days = (tdf["trade_date"].max() - tdf["trade_date"].min()).days
        n_years = max(total_days / 365.0, 0.25)
        cur_val = tdf["dk_value"].dropna().iloc[-1] if not tdf["dk_value"].dropna().empty else float("nan")
        ax_t.set_ylabel(label, fontsize=9)
        ax_t.set_ylim(ax_t.get_ylim())
        ax_t.grid(True, axis="y", alpha=0.2)
        stats = f"switches/yr={switches / n_years:.1f}  signals={n_signals}  cur={cur_val:.3f}"
        ax_t.text(0.99, 0.95, stats, transform=ax_t.transAxes, fontsize=7,
                  ha="right", va="top", bbox=dict(boxstyle="round,pad=0.2",
                  facecolor="white", alpha=0.7))

    ax_bars = fig.add_subplot(gs[-1], sharex=ax_price)
    ax_bars.set_ylabel("Combined", fontsize=9)
    ax_bars.grid(True, axis="y", alpha=0.2)
    ax_bars.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    ax_price.legend(loc="upper left", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print("\n=== 统计汇总 ===")
    for label, trend in results:
        tdf = trend.copy()
        tdf["trade_date"] = pd.to_datetime(tdf["trade_date"])
        switches = (tdf["dk_color"] != tdf["dk_color"].shift(1)).sum()
        total_days = (tdf["trade_date"].max() - tdf["trade_date"].min()).days
        n_years = max(total_days / 365.0, 0.25)
        cur_val = tdf["dk_value"].dropna().iloc[-1] if not tdf["dk_value"].dropna().empty else float("nan")
        last_color = tdf["dk_color"].dropna().iloc[-1] if not tdf["dk_color"].dropna().empty else "N/A"
        print(f"  {label:30s}  {switches / n_years:5.1f} 切换/年  "
              f"当前值={cur_val:7.3f}  颜色={last_color}")


def _parse_compare_specs(raw: str) -> list[tuple[str, DKTrendParams]]:
    specs = [s.strip() for s in raw.split(",") if s.strip()]
    if not specs:
        raise SystemExit("--compare-modes requires at least one mode specification")
    return [_params_from_spec(s) for s in specs]


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot close price with DK trend bars.")
    parser.add_argument("--symbol", required=True, help="Single 6-digit symbol, e.g. 600930")
    parser.add_argument("--start", help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", help="End date, YYYY-MM-DD")
    parser.add_argument("--history", type=int, default=180,
                        help="Keep the latest N valid daily bars after date filtering")
    parser.add_argument("--mode", choices=[m.value for m in TrendMode])
    parser.add_argument("--output", help="PNG output path. Default: data/output/{symbol}_{mode}_dktrend.png")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path, e.g. /path/to/market.duckdb")
    parser.add_argument("--stock-name-cache", help="Override stock name CSV path")
    parser.add_argument("--compare-modes",
                        help="Comma-separated mode specs for visual comparison. "
                             "Format: mode[:param], e.g. long_ma_trend:250,dual_ma_cross:30-120")
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbol = str(args.symbol).strip().zfill(6)
    name_cache_path = (
        Path(args.stock_name_cache).expanduser()
        if args.stock_name_cache
        else resolve_stock_name_cache_path(cfg)
    )
    stock_name = resolve_stock_names([symbol], name_cache_path).get(symbol, symbol)

    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        df = db.read_daily_frame(symbols=[symbol], start=args.start, end=args.end)
    if df.empty:
        raise SystemExit("no daily data found; run scripts/fetch_stock.py first")

    if args.compare_modes:
        compare_specs = _parse_compare_specs(args.compare_modes)
        results: list[tuple[str, pd.DataFrame]] = []
        for label, params in compare_specs:
            trend = compute_dktrend(df, params)
            trend = trend[trend["dk_color"].isin(["red", "green"])].copy()
            if args.history and int(args.history) > 0:
                trend = trend.tail(int(args.history))
            results.append((label, trend))

        suffix = "compare"
        output_path = _resolve_output_path(args.output, cfg, symbol, suffix)
        _plot_comparison(df, results, title=f"{stock_name} ({symbol})", output_path=output_path)
        print(f"已写入 {output_path}")
        return 0

    params = _params(cfg, args.mode)
    mode = str(params.mode.value)
    trend = compute_dktrend(df, params)
    trend = trend[trend["dk_color"].isin(["red", "green"])].copy()
    if args.history and int(args.history) > 0:
        trend = trend.tail(int(args.history))
    if trend.empty:
        raise SystemExit(f"not enough daily data to compute DK trend for {symbol}")

    output_path = _resolve_output_path(args.output, cfg, symbol, mode)
    _plot(trend, title=f"{stock_name} ({symbol})", mode=mode, output_path=output_path)
    print(f"已写入 {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
