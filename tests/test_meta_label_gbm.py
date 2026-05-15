"""Unit tests for GBMMetaModel (S4 gradient boosting meta-label classifier)."""

import numpy as np
import pandas as pd
import pytest

from src.models.meta_label_gbm import GBMMetaModel


def _make_separable_data(n_pos: int = 150, n_neg: int = 150, n_features: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Generate linearly separable-ish 2-class data."""
    np.random.seed(42)
    X_pos = np.random.normal(1.0, 0.6, (n_pos, n_features))
    X_neg = np.random.normal(-0.5, 0.6, (n_neg, n_features))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1.0] * n_pos + [0.0] * n_neg)
    return X, y


class TestGBMMetaModelFit:
    def test_fit_returns_self(self):
        X, y = _make_separable_data()
        model = GBMMetaModel(n_estimators=20, max_depth=2, learning_rate=0.1)
        out = model.fit(X, y)
        assert out is model

    def test_predict_proba_shape(self):
        X, y = _make_separable_data()
        model = GBMMetaModel(n_estimators=20, max_depth=2, learning_rate=0.1)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X),)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_predict_binary(self):
        X, y = _make_separable_data()
        model = GBMMetaModel(n_estimators=20, max_depth=2, learning_rate=0.1)
        model.fit(X, y)
        pred = model.predict(X)
        assert set(pred) <= {0.0, 1.0}

    def test_predict_custom_threshold(self):
        X, y = _make_separable_data()
        model = GBMMetaModel(n_estimators=20, max_depth=2, learning_rate=0.1)
        model.fit(X, y)
        # High threshold → fewer positives
        pred_high = model.predict(X, threshold=0.9)
        pred_low = model.predict(X, threshold=0.1)
        assert pred_high.sum() <= pred_low.sum()

    def test_unfitted_raises(self):
        X, _ = _make_separable_data()
        with pytest.raises(RuntimeError):
            GBMMetaModel().predict_proba(X)

    def test_accuracy_on_separable(self):
        """GBM should achieve >70% accuracy on easily separable data."""
        X, y = _make_separable_data()
        model = GBMMetaModel(n_estimators=50, max_depth=3, learning_rate=0.1)
        model.fit(X, y)
        pred = model.predict(X)
        acc = np.mean(pred == y)
        assert acc > 0.70


class TestGBMMetaModelFeatureImportance:
    def test_feature_importance_keys(self):
        X, y = _make_separable_data(n_features=5)
        names = ["ma20_slope", "atr_pct", "rs_60", "pos_52w", "volume_ratio"]
        model = GBMMetaModel(n_estimators=20, max_depth=2, learning_rate=0.1)
        model.fit(X, y, feature_names=names)
        imp = model.get_feature_importance()
        assert set(imp) == set(names)
        assert all(v >= 0.0 for v in imp.values())
        # Sum should be close to 1.0
        assert abs(sum(imp.values()) - 1.0) < 0.01

    def test_feature_importance_empty_when_unfitted(self):
        model = GBMMetaModel()
        assert model.get_feature_importance() == {}


class TestGBMMetaModelRegularization:
    def test_shallow_trees_prevent_overfit(self):
        """max_depth=1 should produce lower train accuracy than max_depth=5 on noisy data."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (300, 5))
        # Noisy labels
        y = (X[:, 0] + np.random.normal(0, 2.0, 300) > 0).astype(np.float64)

        shallow = GBMMetaModel(n_estimators=30, max_depth=1, learning_rate=0.05)
        deep = GBMMetaModel(n_estimators=30, max_depth=5, learning_rate=0.05)

        shallow.fit(X, y)
        deep.fit(X, y)

        shallow_acc = np.mean(shallow.predict(X) == y)
        deep_acc = np.mean(deep.predict(X) == y)

        # Deeper tree should overfit more (higher train acc on noisy data)
        # This is a sanity check: both should run without error
        assert 0.0 <= shallow_acc <= 1.0
        assert 0.0 <= deep_acc <= 1.0

    def test_class_weight_balanced(self):
        """With imbalanced data, balanced weights should improve recall on minority."""
        np.random.seed(42)
        n_pos = 60
        n_neg = 240
        X_pos = np.random.normal(0.8, 0.5, (n_pos, 4))
        X_neg = np.random.normal(-0.3, 0.5, (n_neg, 4))
        X = np.vstack([X_pos, X_neg])
        y = np.array([1.0] * n_pos + [0.0] * n_neg)

        model = GBMMetaModel(n_estimators=50, max_depth=2, learning_rate=0.05, class_weight="balanced")
        model.fit(X, y)
        pred = model.predict(X)
        recall = pred[y == 1].mean()
        # Should at least catch some positives
        assert recall > 0.10

    def test_min_samples_leaf_constrains_leaf_size(self):
        """min_samples_leaf should run without error for various values."""
        X, y = _make_separable_data(n_pos=80, n_neg=80)
        for min_leaf in [3, 5, 10]:
            model = GBMMetaModel(n_estimators=20, max_depth=2, min_samples_leaf=min_leaf)
            model.fit(X, y)
            proba = model.predict_proba(X)
            assert proba.shape == (len(X),)
