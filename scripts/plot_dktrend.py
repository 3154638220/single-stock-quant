#!/usr/bin/env python
"""Plot DK trend bars from local DuckDB daily data."""

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


def _default_output_path(cfg: dict, symbol: str, mode: str) -> Path:
    out_dir = project_root() / str(cfg.get("paths", {}).get("output_dir", "data/output"))
    return out_dir / f"{symbol}_{mode}_dktrend.png"


def _resolve_output_path(raw: str | None, cfg: dict, symbol: str, mode: str) -> Path:
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = project_root() / path
        return path
    return _default_output_path(cfg, symbol, mode)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot close price with DK trend bars.")
    parser.add_argument("--symbol", required=True, help="Single 6-digit symbol, e.g. 600930")
    parser.add_argument("--start", help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", help="End date, YYYY-MM-DD")
    parser.add_argument("--history", type=int, default=180, help="Keep the latest N valid daily bars after date filtering")
    parser.add_argument("--mode", choices=[m.value for m in TrendMode])
    parser.add_argument("--output", help="PNG output path. Default: data/output/{symbol}_{mode}_dktrend.png")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--duckdb-path", help="Override DuckDB path, e.g. /path/to/market.duckdb")
    parser.add_argument("--stock-name-cache", help="Override stock name CSV path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbol = str(args.symbol).strip().zfill(6)
    params = _params(cfg, args.mode)
    mode = str(params.mode.value)
    name_cache_path = Path(args.stock_name_cache).expanduser() if args.stock_name_cache else resolve_stock_name_cache_path(cfg)
    stock_name = resolve_stock_names([symbol], name_cache_path).get(symbol, symbol)

    with DuckDBManager(config_path=args.config, duckdb_path=args.duckdb_path) as db:
        df = db.read_daily_frame(symbols=[symbol], start=args.start, end=args.end)
    if df.empty:
        raise SystemExit("no daily data found; run scripts/fetch_stock.py first")

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
