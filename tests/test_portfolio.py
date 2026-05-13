"""Tests for portfolio signal ranking and allocation."""
import numpy as np
import pandas as pd

from src.portfolio.allocator import allocate_top_n, apply_constraints
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
