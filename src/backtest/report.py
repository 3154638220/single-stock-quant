"""Generate self-contained HTML backtest reports."""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.performance_panel import bootstrap_sharpe_ci, breakdown_by_regime
from src.backtest.single_stock import SingleStockBacktestResult


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _kpi_card(label: str, value: str, color: str = "#333") -> str:
    return f"""<div style="background:#f8f9fa;border-radius:8px;padding:12px 16px;text-align:center;min-width:100px">
    <div style="font-size:11px;color:#888;margin-bottom:4px">{label}</div>
    <div style="font-size:18px;font-weight:700;color:{color}">{value}</div>
</div>"""


def _equity_chart(result: SingleStockBacktestResult, df: pd.DataFrame, index_ohlcv: pd.DataFrame | None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    dates = pd.to_datetime(df["trade_date"])
    equity = result.daily_returns.add(1.0).cumprod()
    bh = df["close"] / df["close"].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, equity, color="#1f77b4", linewidth=1.2, label="Strategy")
    ax.plot(dates, bh, color="#999", linewidth=0.8, alpha=0.7, label="Buy & Hold")
    if index_ohlcv is not None and not index_ohlcv.empty:
        idx_close = pd.to_numeric(index_ohlcv["close"], errors="coerce")
        if "trade_date" in index_ohlcv.columns:
            idx_dates = pd.to_datetime(index_ohlcv["trade_date"])
        else:
            idx_dates = pd.to_datetime(index_ohlcv.index)
        idx_norm = idx_close / idx_close.iloc[0]
        ax.plot(idx_dates, idx_norm, color="#d62728", linewidth=0.8, alpha=0.6, label="CSI300")
    ax.set_title(f"{result.stock_name} ({result.symbol}) Equity Curve", fontsize=13)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    s = _fig_to_b64(fig)
    plt.close(fig)
    return s


def _drawdown_chart(result: SingleStockBacktestResult) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    equity = result.daily_returns.add(1.0).cumprod()
    running_max = equity.cummax()
    dd = (equity / running_max - 1.0) * 100
    dates = pd.to_datetime(equity.index)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.fill_between(dates, 0, dd, color="#d62728", alpha=0.35, linewidth=0)
    ax.plot(dates, dd, color="#d62728", linewidth=0.6)
    ax.set_title("Drawdown (%)", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    s = _fig_to_b64(fig)
    plt.close(fig)
    return s


def _monthly_heatmap(result: SingleStockBacktestResult) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    returns = result.daily_returns
    monthly = returns.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    table = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
    if table.empty:
        return ""
    table.index = table.index.astype(int)
    table.columns = [int(c) for c in table.columns]
    ann = table.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "")

    fig, ax = plt.subplots(figsize=(10, max(3, len(table) * 0.4)))
    im = ax.imshow(table.values, cmap="RdYlGn", aspect="auto", vmin=-0.2, vmax=0.2)
    for i in range(len(table)):
        for j in range(table.shape[1]):
            v = table.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v*100:.1f}%", ha="center", va="center", fontsize=8,
                        color="#222" if abs(v) < 0.08 else "#fff")
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels([f"{m}月" for m in table.columns], fontsize=8)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table.index, fontsize=8)
    ax.set_title("Monthly Returns Heatmap", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Return")
    fig.tight_layout()
    s = _fig_to_b64(fig)
    plt.close(fig)
    return s


def _trade_distribution(result: SingleStockBacktestResult) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if result.trade_log.empty:
        return ""
    returns = result.trade_log["return"] * 100
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bins = max(10, min(40, len(returns) // 2))
    ax.hist(returns, bins=bins, color="#1f77b4", alpha=0.7, edgecolor="white")
    if len(wins) > 0:
        ax.axvline(wins.mean(), color="green", linestyle="--", linewidth=1, label=f"Mean Win: {wins.mean():.2f}%")
    if len(losses) > 0:
        ax.axvline(losses.mean(), color="red", linestyle="--", linewidth=1, label=f"Mean Loss: {losses.mean():.2f}%")
    ax.set_title(f"Trade Returns Distribution (n={len(returns)}, WR={result.win_rate*100:.1f}%)", fontsize=12)
    ax.set_xlabel("Return (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    s = _fig_to_b64(fig)
    plt.close(fig)
    return s


def _hold_histogram(result: SingleStockBacktestResult) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if result.trade_log.empty:
        return ""
    days = result.trade_log["hold_days"]

    fig, ax = plt.subplots(figsize=(8, 3))
    bins = max(8, min(30, len(days) // 2))
    ax.hist(days, bins=bins, color="#ff7f0e", alpha=0.7, edgecolor="white")
    ax.axvline(days.mean(), color="#d62728", linestyle="--", linewidth=1, label=f"Mean: {days.mean():.1f}d")
    ax.set_title(f"Holding Period Distribution (mean={days.mean():.1f}d)", fontsize=12)
    ax.set_xlabel("Days")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    s = _fig_to_b64(fig)
    plt.close(fig)
    return s


def _fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x*100:.2f}%"


def _fmt_num(x: float, d: int = 2) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{d}f}"


def generate_html_report(
    result: SingleStockBacktestResult,
    ohlcv: pd.DataFrame,
    *,
    index_ohlcv: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
) -> str:
    """Generate a self-contained HTML backtest report.

    Returns the HTML string. If ``output_path`` is given, also writes to disk.
    """
    import matplotlib
    matplotlib.use("Agg")

    lo, hi = bootstrap_sharpe_ci(result.daily_returns.to_numpy(dtype=np.float64))
    returns_arr = result.daily_returns.to_numpy(dtype=np.float64)
    idx_ret_arr = None
    if index_ohlcv is not None and not index_ohlcv.empty:
        from src.backtest.single_stock import _close_to_returns
        idx_ret_arr = _close_to_returns(index_ohlcv, "bench").to_numpy(dtype=np.float64)
    regime = breakdown_by_regime(returns_arr, idx_ret_arr)

    regime_rows = ""
    for label, display in [("bull", "牛市"), ("bear", "熊市"), ("ranging", "震荡")]:
        r = regime.get("regimes", {}).get(label, {})
        n = r.get("n_days", 0)
        sa = r.get("strategy_annualized", float("nan"))
        ia = r.get("index_annualized", float("nan"))
        ex = r.get("excess_annualized", float("nan"))
        sh = r.get("sharpe", float("nan"))
        regime_rows += (
            f"<tr><td>{display}</td><td>{n}</td><td>{_fmt_pct(sa)}</td>"
            f"<td>{_fmt_pct(ia)}</td><td>{_fmt_pct(ex)}</td><td>{_fmt_num(sh)}</td></tr>"
        )

    charts = [
        ("equity_chart", _equity_chart(result, ohlcv, index_ohlcv)),
        ("drawdown_chart", _drawdown_chart(result)),
    ]
    mh = _monthly_heatmap(result)
    if mh:
        charts.append(("monthly_heatmap", mh))
    td = _trade_distribution(result)
    if td:
        charts.append(("trade_distribution", td))
    hh = _hold_histogram(result)
    if hh:
        charts.append(("hold_histogram", hh))

    chart_blocks = "\n".join(
        f'<div style="margin-bottom:24px"><img src="data:image/png;base64,{b64}" style="width:100%;max-width:960px;display:block;margin:0 auto"></div>'
        for _, b64 in charts
    )

    stop_info = f"固定{result.stop_loss_exits} / 追踪{result.trailing_stop_exits} / ATR{result.atr_stop_exits}"
    cost_info = ""
    if result.cost_model:
        cm = result.cost_model
        if cm.get("type") == "symmetric":
            cost_info = f"对称 {cm.get('cost_bps', 0):.0f} bps"
        else:
            cost_info = f"买方{cm.get('buy_fraction', 0)*1e4:.1f}bps / 卖方{cm.get('sell_fraction', 0)*1e4:.1f}bps"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{result.stock_name} ({result.symbol}) 回测报告</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #fff; color: #333; }}
h1 {{ font-size: 22px; margin-bottom: 4px; }}
h2 {{ font-size: 16px; margin: 28px 0 12px; border-bottom: 1px solid #eee; padding-bottom: 6px; }}
.subtitle {{ color: #888; font-size: 13px; margin-bottom: 16px; }}
.kpi-grid {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0 24px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0 16px; }}
th, td {{ border: 1px solid #e0e0e0; padding: 6px 10px; text-align: right; }}
th {{ background: #f5f5f5; font-weight: 600; }}
td:first-child, th:first-child {{ text-align: left; }}
.warn {{ color: #c0392b; font-size: 12px; margin: 4px 0; }}
.footer {{ color: #aaa; font-size: 11px; margin-top: 32px; text-align: center; }}
</style>
</head>
<body>

<h1>{result.stock_name} ({result.symbol}) 回测报告</h1>
<div class="subtitle">区间：{result.period} | 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<h2>核心指标</h2>
<div class="kpi-grid">
{_kpi_card("年化收益", _fmt_pct(result.annualized_return), "#27ae60" if result.annualized_return > 0 else "#c0392b")}
{_kpi_card("夏普比率", _fmt_num(result.sharpe_ratio), "#27ae60" if result.sharpe_ratio > 0.8 else "#333")}
{_kpi_card("Calmar", _fmt_num(result.calmar_ratio))}
{_kpi_card("最大回撤", _fmt_pct(result.max_drawdown), "#c0392b")}
{_kpi_card("胜率", _fmt_pct(result.win_rate))}
{_kpi_card("交易次数", str(result.n_trades))}
{_kpi_card("超额年化", _fmt_pct(result.excess_annualized_return))}
{_kpi_card("信息比率", _fmt_num(result.information_ratio))}
{_kpi_card("Beta", _fmt_num(result.beta_to_benchmark))}
{_kpi_card("平均仓位", _fmt_pct(result.avg_position_fraction))}
{_kpi_card("Sharpe 90%CI", f"{_fmt_num(lo)} ~ {_fmt_num(hi)}")}
{_kpi_card("成本模型", cost_info or "N/A")}
</div>

<h2>止损统计</h2>
<div style="font-size:13px;margin-bottom:12px">
止损退出：{stop_info} | 平均持仓：{result.avg_hold_days:.1f}天 | 单笔平均收益：{_fmt_pct(result.avg_return_per_trade)}
</div>

<h2>市场状态分解</h2>
<table>
<tr><th>状态</th><th>交易日</th><th>策略年化</th><th>基准年化</th><th>超额</th><th>Sharpe</th></tr>
{regime_rows}
</table>

<h2>图表</h2>
{chart_blocks}

<h2>交易记录（最近20笔）</h2>
<table>
<tr><th>买入日</th><th>卖出日</th><th>买入价</th><th>卖出价</th><th>收益</th><th>持仓天数</th><th>退出原因</th></tr>
"""
    if not result.trade_log.empty:
        tail = result.trade_log.tail(20).iloc[::-1]
        for _, row in tail.iterrows():
            html += (
                f"<tr><td>{row['buy_date'].date()}</td><td>{row['sell_date'].date()}</td>"
                f"<td>{row['buy_price']:.2f}</td><td>{row['sell_price']:.2f}</td>"
                f"<td>{row['return']*100:.2f}%</td><td>{int(row['hold_days'])}</td>"
                f"<td>{row['exit_reason']}</td></tr>"
            )
    html += "</table>"

    html += '<div class="footer">Generated by single-stock-quant</div>\n</body>\n</html>'

    if output_path:
        Path(output_path).write_text(html, encoding="utf-8")
    return html
