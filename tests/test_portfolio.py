"""Tests for portfolio signal ranking and allocation."""
import numpy as np
import pandas as pd

from src.portfolio.allocator import allocate_top_n, apply_constraints
from src.portfolio.backtest import build_meta_label_score_panel, run_portfolio_backtest
from src.portfolio.signal_ranker import rank_signals
from src.indicators import DKTrendParams, TrendMode


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

    def test_rank_filters_can_zero_ineligible_rows(self):
        df = _make_identical_daily(["000001"], n_days=140)
        df.loc[df.index[-1], "close"] = 8.0

        scores = rank_signals(df, require_above_ma120=True)

        assert scores["000001"].iloc[-1] == 0.0


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
