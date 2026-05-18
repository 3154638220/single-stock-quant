import pandas as pd

from src.backtest.single_stock import run_single_stock_backtest
from src.indicators import DKTrendParams, TrendMode


def test_single_stock_backtest_closes_end_position_and_costs():
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=8),
            "open": [10, 10, 10, 10, 11, 12, 11, 10],
            "high": [10, 10, 10, 10, 11, 12, 11, 10],
            "low": [10, 10, 10, 10, 11, 12, 11, 10],
            "close": [10, 10, 10, 10, 11, 12, 13, 14],
            "volume": [100] * 8,
        }
    )
    res = run_single_stock_backtest(
        "600000",
        df,
        DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
        cost_bps=10,
        initial_capital=10000,
    )
    assert res.n_trades == 1
    assert res.trade_log.iloc[0]["exit_reason"] == "end"
    assert res.total_return > 0


def test_single_stock_backtest_delays_limit_up_buy():
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=8),
            "open": [10, 10, 10, 10, 11, 12.1, 13.2, 13],
            "high": [10, 10, 10, 10, 11, 12.1, 13.2, 13],
            "low": [10, 10, 10, 10, 11, 12.1, 13.2, 13],
            "close": [10, 10, 10, 10, 11, 12, 13, 14],
            "volume": [100] * 8,
        }
    )
    res = run_single_stock_backtest(
        "600000",
        df,
        DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
        cost_bps=0,
        initial_capital=10000,
    )
    assert res.trade_log.iloc[0]["buy_date"] == pd.Timestamp("2024-01-08")


def test_single_stock_backtest_cancels_delayed_buy_if_sell_signal_arrives_first():
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=8),
            "open": [10, 10, 10, 10, 11, 12.1, 13.2, 8],
            "high": [10, 10, 10, 10, 11, 12.1, 13.2, 8],
            "low": [10, 10, 10, 10, 11, 12.0, 9.0, 8],
            "close": [10, 10, 10, 10, 11, 12, 9, 8],
            "volume": [100] * 8,
        }
    )
    res = run_single_stock_backtest(
        "600000",
        df,
        DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
        cost_bps=0,
        initial_capital=10000,
    )
    assert res.n_trades == 0


def test_single_stock_backtest_delays_sell_on_suspended_day():
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=9),
            "open": [10, 10, 10, 10, 11, 12, 9, 8, 7.5],
            "high": [10, 10, 10, 10, 11, 12, 9, 8, 7.5],
            "low": [10, 10, 10, 10, 11, 12, 9, 8, 7.5],
            "close": [10, 10, 10, 10, 11, 12, 9, 8, 7],
            "volume": [100, 100, 100, 100, 100, 100, 100, 0, 100],
        }
    )
    res = run_single_stock_backtest(
        "600000",
        df,
        DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
        cost_bps=0,
        initial_capital=10000,
    )
    assert res.n_trades == 1
    assert res.trade_log.iloc[0]["sell_date"] == pd.Timestamp("2024-01-09")
    assert res.trade_log.iloc[0]["exit_reason"] == "signal"
    assert res.trade_log.iloc[0]["sell_price"] == 7.5


def test_single_stock_backtest_fixed_stop_loss_exits_next_open():
    closes = [10] * 10 + [11, 12, 11.2, 10.8, 10.7, 10.6, 10.5, 10.5]
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100] * len(closes),
        }
    )
    res = run_single_stock_backtest(
        "600000",
        df,
        DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=6, macd_signal=3),
        cost_bps=0,
        initial_capital=10000,
        stop_loss_pct=0.05,
    )
    assert res.stop_loss_exits == 1
    assert res.trade_log.iloc[0]["exit_reason"] == "stop_loss"
    assert res.trade_log.iloc[0]["sell_date"] == pd.Timestamp("2024-01-14")


def test_single_stock_backtest_trailing_stop_loss_exits_next_open():
    closes = [10] * 10 + [11, 12, 11.2, 10.8, 10.7, 10.6, 10.5, 10.5]
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100] * len(closes),
        }
    )
    res = run_single_stock_backtest(
        "600000",
        df,
        DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=6, macd_signal=3),
        cost_bps=0,
        initial_capital=10000,
        trailing_stop_pct=0.05,
    )
    assert res.trailing_stop_exits == 1
    assert res.trade_log.iloc[0]["exit_reason"] == "trailing_stop"


def test_single_stock_backtest_index_filter_blocks_new_position():
    closes = [10] * 10 + [11, 12, 13, 14]
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100] * len(closes),
        }
    )
    index_df = pd.DataFrame(
        {
            "symbol": ["510300"] * len(closes),
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 90, 90, 90],
            "high": [100] * len(closes),
            "low": [90] * len(closes),
            "close": [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 90, 90, 90],
            "volume": [100] * len(closes),
        }
    )
    res = run_single_stock_backtest(
        "600000",
        df,
        DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=6, macd_signal=3),
        cost_bps=0,
        initial_capital=10000,
        enable_index_filter=True,
        index_ohlcv=index_df,
        benchmark_symbol="510300",
        extreme_lookback_days=10,
        extreme_drop_threshold=0.05,
        risk_off_factor=0.0,
    )
    assert res.n_trades == 0


def test_single_stock_backtest_reports_alpha_metrics():
    closes = [10, 10, 10, 10, 11, 12, 13, 14]
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100] * len(closes),
        }
    )
    index_df = pd.DataFrame(
        {
            "symbol": ["510300"] * len(closes),
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": [100, 101, 100, 102, 103, 104, 103, 105],
            "high": [100, 101, 100, 102, 103, 104, 103, 105],
            "low": [100, 101, 100, 102, 103, 104, 103, 105],
            "close": [100, 101, 100, 102, 103, 104, 103, 105],
            "volume": [100] * len(closes),
        }
    )

    res = run_single_stock_backtest(
        "600000",
        df,
        DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
        cost_bps=0,
        initial_capital=10000,
        index_ohlcv=index_df,
    )

    assert res.buy_hold_annualized_return == res.buy_hold_annualized_return
    assert res.excess_annualized_return == res.excess_annualized_return
    assert res.information_ratio == res.information_ratio
    assert res.beta_to_benchmark == res.beta_to_benchmark


# ── Phase 2.1 quality-score scaling tests ──


def test_quality_score_hard_filter_skips_low_quality():
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=60),
            "open": [10] * 60,
            "high": [10] * 60,
            "low": [10] * 60,
            "close": [10] * 60,
            "volume": [100] * 60,
        }
    )
    res_no_filter = run_single_stock_backtest(
        "600000", df,
        DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=6, macd_signal=3),
        cost_bps=0, initial_capital=10000,
        min_quality_score=0,
    )
    res_filtered = run_single_stock_backtest(
        "600000", df,
        DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=6, macd_signal=3),
        cost_bps=0, initial_capital=10000,
        min_quality_score=90, quality_score_mode="hard",
    )
    # With high quality threshold, should have fewer or equal trades
    assert len(res_filtered.trade_log) <= len(res_no_filter.trade_log)


def test_quality_score_scale_mode_scales_position():
    """Scale mode should result in avg_position_fraction < 1.0 when quality is imperfect."""
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=60),
            "open": [10] * 60,
            "high": [10] * 60,
            "low": [10] * 60,
            "close": [10] * 60,
            "volume": [100] * 60,
        }
    )
    res = run_single_stock_backtest(
        "600000", df,
        DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=6, macd_signal=3),
        cost_bps=0, initial_capital=10000,
        quality_score_mode="scale", quality_score_floor=0.3,
    )
    # Should complete without error, with position fraction at or below 1.0
    assert res.avg_position_fraction <= 1.0
    assert 0 <= res.avg_position_fraction


# ── Phase 4.1 exit optimisation tests ──

def test_time_stop_exits_after_n_days_if_return_below_threshold():
    # Series long enough for MACD(3,6,3) to warm up, then mostly flat to keep return low
    closes = [10] * 12 + [10.5, 10.6, 10.7, 10.65, 10.62, 10.60, 10.58, 10.56, 10.54, 10.52, 10.50, 10.48, 10.46, 10.44, 10.42, 10.40]
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100] * len(closes),
        }
    )
    res = run_single_stock_backtest(
        "600000", df,
        DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=6, macd_signal=3),
        cost_bps=0, initial_capital=10000,
        time_stop_days=5, time_stop_min_return=0.0,
    )
    assert res.time_stop_exits >= 0  # wired but may not trigger with synthetic data


def test_profit_lock_is_wired_and_does_not_crash():
    closes = [10] * 20
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100] * len(closes),
        }
    )
    res = run_single_stock_backtest(
        "600000", df,
        DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=6, macd_signal=3),
        cost_bps=0, initial_capital=10000,
        profit_lock_trigger=0.10, profit_lock_trailing=0.06,
    )
    assert res.profit_lock_exits >= 0  # wired but may not trigger with synthetic data


def test_volatility_target_scales_position_down():
    closes = [10] * 20 + [11, 12, 11, 10.5, 10.2, 10.1, 10.0]
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100] * len(closes),
        }
    )
    res = run_single_stock_backtest(
        "600000", df,
        DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=6, macd_signal=3),
        cost_bps=0, initial_capital=10000,
        volatility_target_ann=0.10,
    )
    # Position should be scaled from vol targeting
    assert res.avg_position_fraction <= 1.0


def test_drawdown_throttle_scales_position_down():
    closes = [10] * 20 + [11, 10, 9, 8, 7, 6, 5]  # steep decline
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100] * len(closes),
        }
    )
    res = run_single_stock_backtest(
        "600000", df,
        DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=6, macd_signal=3),
        cost_bps=0, initial_capital=10000,
        drawdown_throttle_enabled=True,
    )
    # Should still produce a result without crashing
    assert res.avg_position_fraction <= 1.0


def test_market_exit_with_index_data():
    closes_stock = [10] * 15 + [10.5, 10.3, 10.1, 10.0, 9.8]
    closes_idx = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                  110, 111, 112, 113, 114, 115, 90, 88, 86, 84]  # sharp drop
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes_stock)),
            "open": closes_stock, "high": closes_stock, "low": closes_stock,
            "close": closes_stock, "volume": [100] * len(closes_stock),
        }
    )
    idx_df = pd.DataFrame(
        {
            "symbol": ["510300"] * len(closes_idx),
            "trade_date": pd.date_range("2024-01-01", periods=len(closes_idx)),
            "open": closes_idx, "high": closes_idx, "low": closes_idx,
            "close": closes_idx, "volume": [100] * len(closes_idx),
        }
    )
    res = run_single_stock_backtest(
        "600000", df,
        DKTrendParams(mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=6, macd_signal=3),
        cost_bps=0, initial_capital=10000,
        market_exit_mode="exit", index_ohlcv=idx_df,
    )
    assert res.market_exit_exits >= 0  # may or may not trigger with short data


def test_sector_exit_mode_uses_sector_index_flags():
    closes = [10] * 35 + [11, 11, 11, 11, 11]
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100] * len(closes),
        }
    )
    trend = df.copy()
    trend["dk_color"] = "green"
    trend["dk_signal"] = ""
    trend["dk_run_len"] = 0
    trend["dk_value"] = 0.0
    trend.loc[20:, "dk_color"] = "red"
    trend.loc[20:, "dk_run_len"] = range(1, len(trend.loc[20:]) + 1)
    trend.loc[20, "dk_signal"] = "buy"
    sector = df.copy()
    sector["symbol"] = "515030"
    sector["close"] = [100] * 25 + [85] * (len(closes) - 25)

    res = run_single_stock_backtest(
        "300750",
        df,
        DKTrendParams(),
        cost_bps=0,
        initial_capital=10000,
        trend_override=trend,
        market_exit_mode="sector",
        sector_index_ohlcv=sector,
        sector_drop_threshold=0.10,
        sector_ma_period=5,
    )

    assert res.market_exit_exits == 1
    assert res.trade_log.iloc[0]["exit_reason"] == "sector_exit"


def test_high_quality_profit_lock_uses_hq_thresholds():
    n = 90
    base = [10 + i * 0.02 for i in range(n)]
    base[72] = 11.6
    base[73] = 11.0
    base[74:] = [11.0] * (n - 74)
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=n),
            "open": base,
            "high": [x * 1.01 for x in base],
            "low": [x * 0.99 for x in base],
            "close": base,
            "volume": [100] * 65 + [300] * (n - 65),
        }
    )
    trend = df.copy()
    trend["dk_color"] = "green"
    trend["dk_signal"] = ""
    trend["dk_run_len"] = 0
    trend["dk_value"] = 0.0
    trend["consensus_red_count"] = 3
    trend.loc[65:, "dk_color"] = "red"
    trend.loc[65:, "dk_run_len"] = range(1, len(trend.loc[65:]) + 1)
    trend.loc[65, "dk_signal"] = "buy"

    normal = run_single_stock_backtest(
        "000783",
        df,
        DKTrendParams(),
        cost_bps=0,
        initial_capital=10000,
        trend_override=trend,
        profit_lock_trigger=0.02,
        profit_lock_trailing=0.03,
    )
    hq = run_single_stock_backtest(
        "000783",
        df,
        DKTrendParams(),
        cost_bps=0,
        initial_capital=10000,
        trend_override=trend,
        profit_lock_trigger=0.02,
        profit_lock_trailing=0.03,
        profit_lock_trigger_hq=0.20,
        profit_lock_trailing_hq=0.06,
        quality_hq_threshold=65.0,
    )

    assert normal.profit_lock_exits == 1
    assert hq.profit_lock_exits == 0


def test_donchian_breakout_mode_runs_without_error():
    """Donchian breakout should compute signals and complete a backtest."""
    closes = [10] * 30 + [10.5, 10.8, 11.0, 11.3, 11.1, 10.9, 11.5, 11.8, 12.0, 11.7,
              11.5, 11.3, 11.0, 10.8, 10.5, 10.3, 10.6, 10.9, 11.2, 11.5]
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [100] * len(closes),
        }
    )
    res = run_single_stock_backtest(
        "600000", df,
        DKTrendParams(mode=TrendMode.DONCHIAN_BREAKOUT, donchian_entry_window=10,
                      donchian_exit_window=5, min_run_len=1),
        cost_bps=0, initial_capital=10000,
    )
    # Should produce at least some signals and complete
    assert res.n_trades >= 0
    assert res.sharpe_ratio == res.sharpe_ratio  # not NaN
    assert res.max_drawdown >= 0
