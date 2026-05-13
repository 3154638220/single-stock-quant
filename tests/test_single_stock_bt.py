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
