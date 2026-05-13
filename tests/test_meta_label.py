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
