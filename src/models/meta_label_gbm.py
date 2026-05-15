"""Gradient Boosting meta-label classifier (SE4).

Replaces logistic regression with a shallow GBM to handle non-linear feature
relationships while controlling overfitting via max_depth and min_samples_leaf.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class GBMMetaModel:
    """Gradient Boosting meta-label classifier.

    Compared to ``LogisticMetaModel``:
    - Captures non-linear feature interactions
    - Built-in feature importance (mean decrease in impurity)
    - Tree depth control prevents overfitting on small samples
    - Class weighting handles label imbalance

    Parameters
    ----------
    n_estimators:
        Number of boosting stages. Keep low (50-100) for small samples.
    max_depth:
        Maximum tree depth. 2 is recommended to prevent deep interactions.
    learning_rate:
        Shrinks contribution of each tree. Lower = more robust, needs more trees.
    subsample:
        Fraction of samples per tree (stochastic GBM). Improves generalisation.
    min_samples_leaf:
        Minimum samples per leaf. Higher = more regularisation.
    class_weight:
        ``"balanced"`` adjusts weights inversely proportional to class frequencies.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 2,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        min_samples_leaf: int = 5,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self._model: GradientBoostingClassifier | None = None
        self._scaler: StandardScaler | None = None
        self.feature_names_: list[str] = []

    # ------------------------------------------------------------------
    # scikit-learn compatible interface
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> GBMMetaModel:
        """Fit the GBM model on standardised features."""
        n, d = X.shape
        self.feature_names_ = feature_names or [f"f{i}" for i in range(d)]

        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)

        self._model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )

        sample_weight = None
        if self.class_weight == "balanced":
            classes = np.unique(y)
            if len(classes) == 2:
                n_pos = int(np.sum(y == 1))
                n_neg = int(np.sum(y == 0))
                if n_pos > 0 and n_neg > 0:
                    w_pos = len(y) / (2.0 * n_pos)
                    w_neg = len(y) / (2.0 * n_neg)
                    sample_weight = np.where(y == 1, w_pos, w_neg)

        self._model.fit(Xs, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return win probability for each sample. Shape (n_samples,)."""
        if self._model is None or self._scaler is None:
            raise RuntimeError("GBMMetaModel not fitted")
        Xs = self._scaler.transform(X)
        return self._model.predict_proba(Xs)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction at given threshold."""
        return (self.predict_proba(X) >= threshold).astype(np.float64)

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance dict (mean decrease in impurity)."""
        if self._model is None:
            return {}
        return dict(zip(self.feature_names_, self._model.feature_importances_.tolist()))
