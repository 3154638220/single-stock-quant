"""Tests for meta-label model."""
import numpy as np
import pandas as pd
import pytest

from src.models.meta_label import (
    FEATURE_COLUMNS,
    LogisticMetaModel,
    MetaLabelResult,
    _safe_float,
    _sigmoid,
    build_signal_features,
    build_training_samples,
    run_meta_label_wfo,
)
from src.backtest.single_stock import run_single_stock_backtest
from src.indicators import DKTrendParams, TrendMode


def _flat_df(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "trade_date": pd.date_range(start, periods=n),
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [100.0] * n,
        }
    )


def _trend_override(df: pd.DataFrame, buy_idx: int, sell_idx: int | None = None) -> pd.DataFrame:
    trend = df.copy()
    trend["dk_signal"] = ""
    trend["dk_color"] = "green"
    trend["dk_run_len"] = 1
    trend.loc[buy_idx, "dk_signal"] = "buy"
    trend.loc[buy_idx:, "dk_color"] = "red"
    if sell_idx is not None:
        trend.loc[sell_idx, "dk_signal"] = "sell"
        trend.loc[sell_idx:, "dk_color"] = "green"
    return trend


class _ConstantMetaModel:
    def __init__(self, p_win: float):
        self.p_win = p_win

    def predict_proba(self, X):
        return np.full(len(X), self.p_win, dtype=np.float64)


class TestSigmoid:
    def test_sigmoid_range(self):
        x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        p = _sigmoid(x)
        assert np.all(p >= 0) and np.all(p <= 1)
        assert abs(p[2] - 0.5) < 0.01

    def test_sigmoid_monotonic(self):
        x = np.linspace(-5, 5, 100)
        p = _sigmoid(x)
        assert np.all(np.diff(p) >= 0)


class TestLogisticMetaModel:
    def test_fit_and_predict_separable(self):
        np.random.seed(42)
        X = np.vstack([
            np.random.normal(1.0, 0.5, (100, 3)),
            np.random.normal(-1.0, 0.5, (100, 3)),
        ])
        y = np.array([1.0] * 100 + [0.0] * 100)
        model = LogisticMetaModel(l2_penalty=0.01, max_iter=5000)
        model.fit(X, y)
        proba = model.predict_proba(X)
        pred = model.predict(X)
        acc = np.mean(pred == y)
        assert acc > 0.70

    def test_feature_importance_keys(self):
        np.random.seed(42)
        X = np.random.normal(0, 1, (50, 3))
        y = (X[:, 0] > 0).astype(np.float64)
        model = LogisticMetaModel()
        model.fit(X, y, feature_names=["a", "b", "c"])
        imp = model.get_feature_importance()
        assert set(imp) == {"a", "b", "c"}
        assert abs(imp["a"]) > abs(imp["c"])  # first feature most important

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            LogisticMetaModel().predict_proba(np.ones((3, 2)))


class TestBuildSignalFeatures:
    def test_returns_dataframe_with_columns(self):
        closes = list(range(100, 250))
        df = _flat_df([float(c) for c in closes])
        feats = build_signal_features(df)
        for c in FEATURE_COLUMNS:
            assert c in feats.columns

    def test_no_index_ohlcv_uses_stock_ret(self):
        closes = list(range(100, 250))
        df = _flat_df([float(c) for c in closes])
        feats = build_signal_features(df)
        assert "rs_60" in feats.columns

    def test_phase11_features_present_and_bounded(self):
        n = 320
        closes = [100.0 + i * 0.15 + 2.0 * np.sin(i / 8.0) for i in range(n)]
        df = _flat_df(closes)
        df["volume"] = [1000.0 + (i % 20) * 15.0 for i in range(n)]
        df["turnover_rate"] = [1.0 + (i % 60) / 60.0 for i in range(n)]
        index_df = _flat_df([100.0 + i * 0.08 for i in range(n)])

        feats = build_signal_features(df, index_ohlcv=index_df)
        expected = {
            "pos_52w",
            "pv_diverge",
            "trend_consistency_20",
            "macd_hist_dir",
            "turnover_rank_60",
            "beta_120",
            "close_accel_10",
            "vol_price_corr_20",
        }
        assert expected <= set(FEATURE_COLUMNS)
        assert expected <= set(feats.columns)
        assert feats["pos_52w"].dropna().between(0.0, 1.0).all()
        assert feats["trend_consistency_20"].dropna().between(0.0, 1.0).all()
        assert feats["turnover_rank_60"].dropna().between(0.0, 1.0).all()
        assert np.isfinite(feats["beta_120"].iloc[-1])


class TestBuildTrainingSamples:
    def test_profit_label_binary(self):
        closes = [10.0] * 40 + [12.0] * 20 + [11.0] * 40
        df = _flat_df(closes)
        sig_dates = pd.DatetimeIndex([df["trade_date"].iloc[5], df["trade_date"].iloc[35]])
        X, y, dates = build_training_samples(df, sig_dates, forward_return_days=20)
        assert len(y) >= 1
        assert set(y) <= {0.0, 1.0}

    def test_risk_reward_label_binary(self):
        closes = [10.0] * 30 + [11.0] * 10 + [9.0] * 10 + [10.0] * 50
        df = _flat_df(closes)
        sig_dates = pd.DatetimeIndex([df["trade_date"].iloc[5]])
        X, y, dates = build_training_samples(df, sig_dates, label_type="risk_reward")
        assert len(y) >= 1
        assert set(y) <= {0.0, 1.0}

    def test_signal_too_close_to_end_skipped(self):
        closes = [10.0] * 100
        df = _flat_df(closes)
        sig_dates = pd.DatetimeIndex([df["trade_date"].iloc[95]])
        X, y, dates = build_training_samples(df, sig_dates, forward_return_days=20)
        assert len(y) == 0


class TestMetaLabelWFO:
    def test_wfo_returns_results(self):
        n = 800
        closes = [10.0 + 0.01 * i for i in range(n)]
        df = _flat_df(closes, start="2022-01-01")
        # Simple signal dates: every 5 bars starting at bar 30
        sig_dates = pd.DatetimeIndex(df["trade_date"].iloc[30:n - 30:5])
        results = run_meta_label_wfo(
            df, sig_dates, train_days=300, oos_days=100,
            min_signals_per_fold=3, l2_penalty=0.1,
        )
        assert len(results) >= 1
        for r in results:
            assert isinstance(r, MetaLabelResult)


class TestMetaLabelBacktestIntegration:
    def test_hard_filter_skips_low_p_win_buy(self):
        df = _flat_df([10.0, 10.0, 10.5, 11.0, 11.2, 11.0, 10.8, 10.7])
        trend = _trend_override(df, buy_idx=1, sell_idx=5)
        params = DKTrendParams(mode=TrendMode.MACD_CROSS)

        baseline = run_single_stock_backtest(
            "600000",
            df,
            params,
            cost_bps=0,
            initial_capital=10000,
            trend_override=trend,
        )
        filtered = run_single_stock_backtest(
            "600000",
            df,
            params,
            cost_bps=0,
            initial_capital=10000,
            trend_override=trend,
            meta_model=_ConstantMetaModel(0.40),
            meta_label_mode="hard",
            meta_label_threshold=0.55,
        )

        assert baseline.n_trades == 1
        assert filtered.n_trades == 0
