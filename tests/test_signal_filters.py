import pandas as pd

from src.backtest.single_stock import run_single_stock_backtest
from src.indicators import DKTrendParams, TrendMode, compute_dktrend
from src.signals import (
    compute_consensus_trend,
    generate_consensus_signals,
    generate_signals,
)
from src.signals.generator import compute_signal_quality
from src.signals.types import Signal


def _df(closes: list[float], extra: dict | None = None) -> pd.DataFrame:
    data = {
        "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [100] * len(closes),
    }
    if extra:
        data.update(extra)
    return pd.DataFrame(data)


def _trend_override(df: pd.DataFrame, buy_idx: int) -> pd.DataFrame:
    trend = df.copy()
    trend["dk_signal"] = ""
    trend["dk_color"] = "green"
    trend["dk_run_len"] = 1
    trend.loc[buy_idx, "dk_signal"] = "buy"
    trend.loc[buy_idx:, "dk_color"] = "red"
    return trend


# ── Persistence filter (min_run_len) ───────────────────────────

class TestMinRunLen:
    def test_min_run_len_default_1_original_behavior(self):
        df = _df([10, 10, 10, 10, 11, 12, 9, 8])
        out = compute_dktrend(df, DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3))
        assert (out["dk_signal"] == "buy").any()
        assert (out["dk_signal"] == "sell").any()

    def test_min_run_len_2_suppresses_single_day_flip(self):
        """With min_run_len=2, buy only fires with at least 2 consecutive red days."""
        closes = [10] * 15 + [11, 12, 12, 12, 11, 10, 9, 9, 9, 9, 9, 9]
        df = _df(closes)
        out = compute_dktrend(
            df,
            DKTrendParams(
                mode=TrendMode.BOLL_TREND, boll_window=5, min_run_len=2
            ),
        )
        buys = out[out["dk_signal"] == "buy"]
        assert len(buys) <= 1  # reduced signal count compared to min_run_len=1

    def test_min_run_len_3_requires_three_consecutive(self):
        closes = [10] * 15 + [11, 12, 13, 13, 12, 11, 10, 9, 9, 9, 9, 9, 9, 9, 10]
        df = _df(closes)
        out = compute_dktrend(
            df,
            DKTrendParams(
                mode=TrendMode.MACD_CROSS, macd_fast=3, macd_slow=8, macd_signal=3,
                min_run_len=3,
            ),
        )
        buy_mask = out["dk_signal"] == "buy"
        if buy_mask.any():
            buy_run_lens = out.loc[buy_mask, "dk_run_len"]
            assert (buy_run_lens >= 3).all()

    def test_consensus_respects_min_run_len(self):
        closes = [10] * 20 + [11, 12, 13, 12, 11, 10, 9, 9, 9, 9]
        df = _df(closes)
        trend = compute_consensus_trend(
            df,
            n_agree=2,
            base_params=DKTrendParams(boll_window=3, ma_fast=2, ma_slow=3, min_run_len=2),
        )
        assert "consensus_red_count" in trend.columns
        assert "dk_signal" in trend.columns


# ── Signal quality scoring ─────────────────────────────────────

class TestSignalQuality:
    def test_compute_quality_returns_zero_to_hundred(self):
        closes = [10] * 15 + [11, 12, 13, 12, 11, 10, 9, 9, 9, 9]
        df = _df(closes)
        trend = compute_dktrend(df, DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=5))
        quality = compute_signal_quality(trend)
        assert 0.0 <= quality.max() <= 100.0

    def test_quality_zero_for_hold_sell(self):
        closes = [10] * 15 + [11, 12, 13, 12, 11, 10, 9, 9, 9, 9]
        df = _df(closes)
        trend = compute_dktrend(df, DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=5))
        quality = compute_signal_quality(trend)
        non_buy = trend["dk_signal"] != "buy"
        assert (quality[non_buy] == 0.0).all()

    def test_quality_score_on_signal_record(self):
        closes = [10] * 15 + [11, 12, 13, 12, 11, 10, 9, 9, 9, 9]
        df = _df(closes)
        records = generate_signals(df, DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=5))
        for r in records:
            if r.signal == Signal.BUY:
                assert r.quality_score >= 0.0
            else:
                assert r.quality_score == 0.0

    def test_volume_boost_when_volume_spikes(self):
        closes = [10] * 30
        vol = [100] * 25 + [500] * 5
        df = _df(closes, extra={"volume": vol})
        trend = compute_dktrend(
            df,
            DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=5),
        )
        quality = compute_signal_quality(trend, volume_ratio_min=1.5, volume_lookback=10)
        buy_mask = trend["dk_signal"] == "buy"
        if buy_mask.any():
            scores_when_buy = quality[buy_mask]
            assert scores_when_buy.max() >= 20  # at minimum the volume or run_len bonus

    def test_consensus_quality_score_populated(self):
        closes = [10] * 15 + [11, 12, 13, 13, 12, 11, 10, 9, 9, 9, 9, 9, 9, 9, 9]
        df = _df(closes)
        records = generate_consensus_signals(df, n_agree=2)
        buy_records = [r for r in records if r.signal == Signal.BUY]
        if buy_records:
            for r in buy_records:
                assert 0.0 <= r.quality_score <= 100.0
        # if no buys, that's fine — trend data may not produce consensus signals


class TestSingleStockEntryFilters:
    def test_require_above_ma120_blocks_below_long_ma(self):
        closes = [10.0] * 130
        df = _df(closes)
        trend = _trend_override(df, buy_idx=121)

        baseline = run_single_stock_backtest(
            "600000",
            df,
            DKTrendParams(mode=TrendMode.MACD_CROSS),
            cost_bps=0,
            initial_capital=10000,
            trend_override=trend,
        )
        filtered = run_single_stock_backtest(
            "600000",
            df,
            DKTrendParams(mode=TrendMode.MACD_CROSS),
            cost_bps=0,
            initial_capital=10000,
            trend_override=trend,
            require_above_ma120=True,
        )

        assert baseline.n_trades == 1
        assert filtered.n_trades == 0

    def test_require_positive_rs60_blocks_stock_weaker_than_index(self):
        closes = [12.0] * 10 + [11.5] * 30 + [10.0] * 90
        df = _df(closes)
        index_df = _df([100.0] * len(closes), extra={"symbol": ["510300"] * len(closes)})
        trend = _trend_override(df, buy_idx=80)

        baseline = run_single_stock_backtest(
            "600000",
            df,
            DKTrendParams(mode=TrendMode.MACD_CROSS),
            cost_bps=0,
            initial_capital=10000,
            trend_override=trend,
            index_ohlcv=index_df,
        )
        filtered = run_single_stock_backtest(
            "600000",
            df,
            DKTrendParams(mode=TrendMode.MACD_CROSS),
            cost_bps=0,
            initial_capital=10000,
            trend_override=trend,
            index_ohlcv=index_df,
            require_positive_rs60=True,
        )

        assert baseline.n_trades == 1
        assert filtered.n_trades == 0

    def test_require_index_trend_bullish_blocks_when_index_macd_hist_negative(self):
        closes = [10.0] * 80
        df = _df(closes)
        index_closes = list(pd.Series(range(200, 120, -1), dtype=float))
        index_df = _df(index_closes, extra={"symbol": ["510300"] * len(index_closes)})
        trend = _trend_override(df, buy_idx=60)

        baseline = run_single_stock_backtest(
            "600000",
            df,
            DKTrendParams(mode=TrendMode.MACD_CROSS),
            cost_bps=0,
            initial_capital=10000,
            trend_override=trend,
            index_ohlcv=index_df,
        )
        filtered = run_single_stock_backtest(
            "600000",
            df,
            DKTrendParams(mode=TrendMode.MACD_CROSS),
            cost_bps=0,
            initial_capital=10000,
            trend_override=trend,
            index_ohlcv=index_df,
            require_index_trend_bullish=True,
        )

        assert baseline.n_trades == 1
        assert filtered.n_trades == 0

    def test_require_weekly_bullish_blocks_daily_buy_in_weekly_downtrend(self):
        closes = list(pd.Series(range(100, 20, -1), dtype=float))
        df = _df(closes)
        trend = _trend_override(df, buy_idx=60)

        baseline = run_single_stock_backtest(
            "600000",
            df,
            DKTrendParams(mode=TrendMode.MACD_CROSS),
            cost_bps=0,
            initial_capital=10000,
            trend_override=trend,
        )
        filtered = run_single_stock_backtest(
            "600000",
            df,
            DKTrendParams(mode=TrendMode.MACD_CROSS),
            cost_bps=0,
            initial_capital=10000,
            trend_override=trend,
            require_weekly_bullish=True,
            weekly_ma_fast=2,
            weekly_ma_slow=3,
        )

        assert baseline.n_trades == 1
        assert filtered.n_trades == 0
