"""Tests for portfolio signal ranking and allocation."""
import numpy as np
import pandas as pd

from src.indicators import DKTrendParams, TrendMode
from src.portfolio.allocator import allocate_top_n, apply_constraints
from src.portfolio.attribution import (
    calibrate_scores_by_forward_returns,
    compute_candidate_forward_return_breakdown,
    compute_score_forward_return_attribution,
    summarize_score_monotonicity,
)
from src.portfolio.backtest import build_meta_label_score_panel, run_portfolio_backtest
from src.portfolio.signal_ranker import rank_signals


def _make_daily(symbols: list[str], n_days: int = 100, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    rows = []
    for s in symbols:
        base = 10.0 + np.random.randn() * 2
        r = np.random.randn(n_days) * 0.02
        closes = base * np.cumprod(1 + r)
        for i, c in enumerate(closes):
            rows.append({
                "trade_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "symbol": s,
                "open": c * 0.99,
                "high": c * 1.02,
                "low": c * 0.98,
                "close": c,
                "volume": np.random.uniform(1e6, 1e7),
            })
    return pd.DataFrame(rows)


def _make_identical_daily(symbols: list[str], n_days: int = 140) -> pd.DataFrame:
    rows = []
    closes = 10.0 + np.linspace(0, 2.0, n_days)
    for s in symbols:
        for i, c in enumerate(closes):
            rows.append({
                "trade_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "symbol": s,
                "open": c * 0.99,
                "high": c * 1.02,
                "low": c * 0.98,
                "close": c,
                "volume": 2e7,
            })
    return pd.DataFrame(rows)


class TestSignalRanker:
    def test_rank_signals_returns_wide(self):
        df = _make_daily(["000001", "000002"], n_days=100)
        scores = rank_signals(df)
        assert isinstance(scores, pd.DataFrame)
        assert scores.shape[1] >= 1
        assert "000001" in scores.columns

    def test_scores_in_valid_range(self):
        df = _make_daily(["000001", "000002"], n_days=100)
        scores = rank_signals(df)
        assert scores.min().min() >= 0.0
        assert scores.max().max() <= 100.0

    def test_rank_signals_with_index(self):
        df = _make_daily(["000001"], n_days=100)
        idx_df = _make_daily(["510300"], n_days=100)
        scores = rank_signals(df, index_ohlcv=idx_df)
        assert isinstance(scores, pd.DataFrame)

    def test_rank_signals_handles_short_data(self):
        df = _make_daily(["000001"], n_days=10)
        scores = rank_signals(df)
        assert isinstance(scores, pd.DataFrame)

    def test_meta_label_scores_affect_cross_sectional_rank(self):
        df = _make_identical_daily(["000001", "000002"], n_days=140)
        dates = pd.to_datetime(sorted(df["trade_date"].unique()))
        meta = pd.DataFrame({"000001": 0.45, "000002": 0.80}, index=dates)

        scores = rank_signals(df, meta_label_scores=meta)

        assert scores["000002"].iloc[-1] > scores["000001"].iloc[-1]

    def test_meta_priority_profile_increases_meta_label_spread(self):
        df = _make_identical_daily(["000001", "000002"], n_days=140)
        dates = pd.to_datetime(sorted(df["trade_date"].unique()))
        meta = pd.DataFrame({"000001": 0.45, "000002": 0.80}, index=dates)

        balanced = rank_signals(df, meta_label_scores=meta, ranking_profile="balanced")
        meta_priority = rank_signals(df, meta_label_scores=meta, ranking_profile="meta_priority")

        balanced_spread = balanced["000002"].iloc[-1] - balanced["000001"].iloc[-1]
        meta_spread = meta_priority["000002"].iloc[-1] - meta_priority["000001"].iloc[-1]
        assert meta_spread > balanced_spread

    def test_rank_filters_can_zero_ineligible_rows(self):
        df = _make_identical_daily(["000001"], n_days=140)
        df.loc[df.index[-1], "close"] = 8.0

        scores = rank_signals(df, require_above_ma120=True)

        assert scores["000001"].iloc[-1] == 0.0

    def test_dk_meta_profile_requires_red_trend_candidate(self):
        dates = pd.date_range("2024-01-01", periods=160, freq="D")
        rows = []
        for sym, slope in [("000001", 0.01), ("000002", -0.01)]:
            for i, dt in enumerate(dates):
                close = 10.0 * (1.0 + slope) ** i
                rows.append(
                    {
                        "trade_date": dt,
                        "symbol": sym,
                        "open": close,
                        "high": close * 1.01,
                        "low": close * 0.99,
                        "close": close,
                        "volume": 2e7,
                    }
                )
        df = pd.DataFrame(rows)
        meta = pd.DataFrame({"000001": 0.70, "000002": 0.90}, index=dates)

        scores = rank_signals(
            df,
            meta_label_scores=meta,
            ranking_profile="dk_meta",
            dk_params=DKTrendParams(mode=TrendMode.MA_CROSS, ma_fast=3, ma_slow=10),
        )

        assert scores["000001"].iloc[-1] > 0.0
        assert scores["000002"].iloc[-1] == 0.0

    def test_dk_fresh_meta_profile_excludes_stale_red_trend(self):
        dates = pd.date_range("2024-01-01", periods=180, freq="D")
        old_red = 10.0 * np.cumprod(np.repeat(1.01, len(dates)))
        base = np.concatenate(
            [
                np.linspace(12.0, 8.0, 160),
                8.0 * np.cumprod(np.repeat(1.05, 20)),
            ]
        )
        rows = []
        for sym, closes in [("000001", old_red), ("000002", base)]:
            for dt, close in zip(dates, closes, strict=True):
                rows.append(
                    {
                        "trade_date": dt,
                        "symbol": sym,
                        "open": close,
                        "high": close * 1.01,
                        "low": close * 0.99,
                        "close": close,
                        "volume": 2e7,
                    }
                )
        df = pd.DataFrame(rows)
        meta = pd.DataFrame({"000001": 0.90, "000002": 0.70}, index=dates)

        scores = rank_signals(
            df,
            meta_label_scores=meta,
            ranking_profile="dk_fresh_meta",
            dk_params=DKTrendParams(mode=TrendMode.MA_CROSS, ma_fast=3, ma_slow=10),
        )

        assert scores["000001"].iloc[-1] == 0.0
        assert scores["000002"].iloc[-1] > 0.0

    def test_symbol_exclusion_guardrail_zeroes_scores(self):
        df = _make_identical_daily(["000001", "000002"], n_days=140)

        scores = rank_signals(df, exclude_symbols=["1"])

        assert scores["000001"].max() == 0.0
        assert scores["000002"].max() > 0.0

    def test_symbol_greylist_guardrail_scales_scores(self):
        df = _make_identical_daily(["000001", "000002"], n_days=140)
        baseline = rank_signals(df)

        guarded = rank_signals(df, greylist_symbols=["000002"], greylist_score_scale=0.25)

        assert np.isclose(guarded["000002"].iloc[-1], baseline["000002"].iloc[-1] * 0.25)
        assert np.isclose(guarded["000001"].iloc[-1], baseline["000001"].iloc[-1])

    def test_rolling_greylist_profile_includes_params(self):
        from src.portfolio.signal_ranker import _ranking_profile_weights
        w = _ranking_profile_weights("dk_rolling_greylist")
        assert w["rolling_greylist_lookback"] == 126
        assert w["rolling_greylist_horizon"] == 5
        assert w["rolling_greylist_threshold"] == -0.01
        assert w["rolling_greylist_scale"] == 0.0
        assert w["rolling_greylist_min_samples"] == 5
        assert w["require_dk_red"] is True

    def test_rolling_greylist_positive_returns_do_not_trigger(self):
        symbols = ["000001", "000002", "000003"]
        n_days = 200
        rows = []
        for s in symbols:
            base = 50.0 if s == "000002" else 30.0
            r = np.random.randn(n_days) * 0.02
            closes = base * np.cumprod(1 + r)
            for i, c in enumerate(closes):
                rows.append({
                    "trade_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "symbol": s,
                    "open": c * 0.99,
                    "high": c * 1.02,
                    "low": c * 0.98,
                    "close": c,
                    "volume": 2e7,
                })
        df = pd.DataFrame(rows)

        scores = rank_signals(
            df,
            rolling_greylist_lookback=126,
            rolling_greylist_horizon=5,
            rolling_greylist_threshold=-0.01,
            rolling_greylist_scale=0.0,
            rolling_greylist_min_samples=5,
        )
        assert not scores.isna().all(axis=None)

    def test_rolling_greylist_suppresses_negative_symbol(self):
        symbols = ["000001", "000002"]
        n_days = 200
        rows = []
        for s in symbols:
            if s == "000002":
                r = np.linspace(0, -0.3, n_days)
            else:
                r = np.random.randn(n_days) * 0.02
            base = 10.0
            closes = base * np.cumprod(1 + r)
            for i, c in enumerate(closes):
                rows.append({
                    "trade_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "symbol": s,
                    "open": c * 0.99,
                    "high": c * 1.02,
                    "low": c * 0.98,
                    "close": c,
                    "volume": 2e7,
                })
        df = pd.DataFrame(rows)

        scores = rank_signals(
            df,
            rolling_greylist_lookback=126,
            rolling_greylist_horizon=5,
            rolling_greylist_threshold=-0.01,
            rolling_greylist_scale=0.0,
            rolling_greylist_min_samples=5,
        )
        assert scores["000001"].sum() > 0
        recent = scores.iloc[-20:]
        assert (recent["000002"] == 0.0).all(), "consistently negative symbol should be greylisted in recent period"

    def test_rolling_greylist_scale_mode_reduces_not_zeroes(self):
        symbols = ["000001"]
        n_days = 200
        rows = []
        r = np.linspace(0, -0.15, n_days)
        base = 10.0
        closes = base * np.cumprod(1 + r)
        for i, c in enumerate(closes):
            rows.append({
                "trade_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "symbol": "000001",
                "open": c * 0.99,
                "high": c * 1.02,
                "low": c * 0.98,
                "close": c,
                "volume": 2e7,
            })
        df = pd.DataFrame(rows)

        baseline = rank_signals(df)
        scaled = rank_signals(
            df,
            rolling_greylist_lookback=126,
            rolling_greylist_horizon=5,
            rolling_greylist_threshold=-0.01,
            rolling_greylist_scale=0.50,
            rolling_greylist_min_samples=5,
        )
        recent_base = baseline["000001"].iloc[-20:]
        recent_scaled = scaled["000001"].iloc[-20:]
        assert (recent_scaled <= recent_base).all()


class TestAllocator:
    def test_allocate_top_n_output_shape(self):
        df = _make_daily(["000001", "000002", "000003", "000004", "000005", "000006"], n_days=100)
        scores = rank_signals(df)
        weights = allocate_top_n(scores, n_top=3)
        assert weights.shape == scores.shape
        assert weights.max().max() <= 0.25
        for dt in weights.index:
            assert weights.loc[dt].sum() <= 1.0 + 1e-10

    def test_allocate_respects_n_top(self):
        df = _make_daily(["000001", "000002", "000003", "000004", "000005", "000006"], n_days=100)
        scores = rank_signals(df)
        weights = allocate_top_n(scores, n_top=2)
        for dt in weights.index:
            assert (weights.loc[dt] > 0).sum() <= 2

    def test_constraints_enforce_max_per_stock(self):
        df = _make_daily([f"{i:06d}" for i in range(10)], n_days=60)
        scores = rank_signals(df)
        weights = allocate_top_n(scores, n_top=5, max_per_stock=0.20)
        constrained = apply_constraints(weights, max_positions=5, max_per_stock=0.20)
        assert constrained.max().max() <= 0.20

    def test_constraints_enforce_max_positions(self):
        df = _make_daily([f"{i:06d}" for i in range(10)], n_days=60)
        scores = rank_signals(df)
        weights = allocate_top_n(scores, n_top=5)
        constrained = apply_constraints(weights, max_positions=3)
        for dt in constrained.index:
            assert (constrained.loc[dt] > 0).sum() <= 3


class TestPortfolioBacktest:
    def test_run_portfolio_backtest_accepts_meta_scores(self):
        df = _make_identical_daily(["000001", "000002", "000003"], n_days=150)
        dates = pd.to_datetime(sorted(df["trade_date"].unique()))
        meta = pd.DataFrame(
            {"000001": 0.80, "000002": 0.55, "000003": 0.45},
            index=dates,
        )

        result = run_portfolio_backtest(
            df,
            n_top=2,
            max_per_stock=0.5,
            min_volume_amount=0,
            meta_label_scores=meta,
            min_meta_score=0.50,
            ranking_profile="meta_priority",
        )

        assert result["backtest"] is not None
        assert "summary" in result
        assert "000003" in result["weights"].columns
        assert result["weights"]["000003"].max() == 0.0

    def test_build_meta_label_score_panel_returns_wide_probabilities(self):
        df = _make_identical_daily(["000001", "000002"], n_days=90)
        scores = build_meta_label_score_panel(
            df,
            DKTrendParams(mode=TrendMode.BOLL_TREND),
            min_train_days=40,
            refit_every=10,
            min_samples=1,
        )

        assert list(scores.columns) == ["000001", "000002"]
        assert scores.min().min() >= 0.0
        assert scores.max().max() <= 1.0

    def test_symbol_exclusion_guardrail_removes_portfolio_weight(self):
        df = _make_identical_daily(["000001", "000002", "000003"], n_days=150)
        dates = pd.to_datetime(sorted(df["trade_date"].unique()))
        meta = pd.DataFrame(
            {"000001": 0.90, "000002": 0.80, "000003": 0.70},
            index=dates,
        )

        result = run_portfolio_backtest(
            df,
            n_top=2,
            max_per_stock=0.5,
            min_volume_amount=0,
            meta_label_scores=meta,
            ranking_profile="meta_priority",
            exclude_symbols=["000001"],
        )

        assert result["weights"]["000001"].max() == 0.0
        assert result["weights"][["000002", "000003"]].max().max() > 0.0


class TestPortfolioAttribution:
    def test_score_forward_return_attribution_detects_top_bucket_edge(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        rows = []
        for sym, slope in [("000001", 0.02), ("000002", -0.01)]:
            for i, dt in enumerate(dates):
                px = 10.0 * (1.0 + slope) ** i
                rows.append(
                    {
                        "trade_date": dt,
                        "symbol": sym,
                        "open": px,
                        "high": px * 1.01,
                        "low": px * 0.99,
                        "close": px,
                        "volume": 1_000_000,
                    }
                )
        daily = pd.DataFrame(rows)
        scores = pd.DataFrame({"000001": 90.0, "000002": 10.0}, index=dates)

        attribution = compute_score_forward_return_attribution(
            daily,
            scores,
            horizons=(1,),
            n_quantiles=2,
        )
        summary = summarize_score_monotonicity(attribution)

        assert set(attribution["bucket"]) == {"Q1", "Q2"}
        assert summary["top_minus_bottom"].iloc[0] > 0
        assert bool(summary["is_monotonic"].iloc[0])

    def test_forward_return_calibration_uses_historical_bucket_edge(self):
        dates = pd.date_range("2024-01-01", periods=80, freq="D")
        rows = []
        for sym, slope in [("000001", -0.01), ("000002", 0.02)]:
            for i, dt in enumerate(dates):
                px = 10.0 * (1.0 + slope) ** i
                rows.append(
                    {
                        "trade_date": dt,
                        "symbol": sym,
                        "open": px,
                        "high": px * 1.01,
                        "low": px * 0.99,
                        "close": px,
                        "volume": 1_000_000,
                    }
                )
        daily = pd.DataFrame(rows)
        raw_scores = pd.DataFrame({"000001": 90.0, "000002": 10.0}, index=dates)

        calibrated = calibrate_scores_by_forward_returns(
            daily,
            raw_scores,
            horizon=5,
            lookback_days=40,
            n_quantiles=2,
            min_observations=20,
            calibration_strength=0.70,
        )

        assert calibrated["000002"].iloc[-1] > calibrated["000001"].iloc[-1]

    def test_dk_calibrated_meta_profile_is_available(self):
        df = _make_identical_daily(["000001", "000002"], n_days=140)
        scores = rank_signals(
            df,
            ranking_profile="dk_calibrated_meta",
            dk_params=DKTrendParams(mode=TrendMode.MA_CROSS, ma_fast=3, ma_slow=10),
        )

        assert isinstance(scores, pd.DataFrame)
        assert scores.min().min() >= 0.0
        assert scores.max().max() <= 100.0

    def test_candidate_breakdown_flags_weak_symbols(self):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        rows = []
        for sym, slope in [("000001", 0.02), ("000002", -0.01)]:
            for i, dt in enumerate(dates):
                px = 10.0 * (1.0 + slope) ** i
                rows.append(
                    {
                        "trade_date": dt,
                        "symbol": sym,
                        "open": px,
                        "high": px * 1.01,
                        "low": px * 0.99,
                        "close": px,
                        "volume": 1_000_000,
                    }
                )
        daily = pd.DataFrame(rows)
        scores = pd.DataFrame({"000001": 70.0, "000002": 70.0}, index=dates)

        breakdown = compute_candidate_forward_return_breakdown(
            daily,
            scores,
            horizons=(5,),
            group_by=("symbol",),
        )
        by_symbol = breakdown.set_index("symbol")

        assert by_symbol.loc["000001", "mean_forward_return"] > 0
        assert by_symbol.loc["000002", "mean_forward_return"] < 0

    def test_candidate_breakdown_supports_market_regime_and_industry(self):
        daily = _make_identical_daily(["000001"], n_days=100)
        dates = pd.to_datetime(sorted(daily["trade_date"].unique()))
        scores = pd.DataFrame({"000001": 80.0}, index=dates)
        index_df = daily.copy()
        index_df["symbol"] = "510300"

        breakdown = compute_candidate_forward_return_breakdown(
            daily,
            scores,
            horizons=(1,),
            index_ohlcv=index_df,
            industry_map={"000001": "bank"},
            group_by=("market_regime", "industry"),
        )

        assert "bull" in set(breakdown["market_regime"])
        assert set(breakdown["industry"]) == {"bank"}
