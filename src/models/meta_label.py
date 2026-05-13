"""Meta-labeling: predict whether a BUY signal is worth executing.

Lightweight logistic-regression model trained on per-signal features.
All evaluation is WFO-based — no random train/test split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MetaLabelResult:
    """Output from meta-label training/evaluation for one fold."""

    fold: int
    train_n: int
    test_n: int
    train_precision: float = float("nan")
    test_precision: float = float("nan")
    test_recall: float = float("nan")
    test_n_pred_positive: int = 0
    feature_importance: dict[str, float] = field(default_factory=dict)
    coef: list[float] = field(default_factory=list)
    intercept: float = float("nan")


def _safe_float(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.zeros_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    out[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))
    return out


def build_signal_features(
    ohlcv: pd.DataFrame,
    *,
    index_ohlcv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build feature matrix for every row in *ohlcv*.

    Returns a DataFrame with feature columns indexed by position.
    """
    df = ohlcv.copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    out = pd.DataFrame(index=df.index)

    # Moving average features
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    out["ma20_slope"] = (ma20 / ma20.shift(5) - 1.0) * 100
    out["ma60_slope"] = (ma60 / ma60.shift(5) - 1.0) * 100
    out["close_above_ma20"] = (close > ma20).astype(float)
    out["close_above_ma60"] = (close > ma60).astype(float)

    # Volatility (ATR %)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    out["atr_pct"] = atr / close * 100
    out["atr_pct_rank"] = (
        out["atr_pct"]
        .rolling(120, min_periods=60)
        .apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)
    )

    # Volume ratio
    avg_vol_20 = volume.rolling(20, min_periods=20).mean()
    out["volume_ratio_20"] = volume / avg_vol_20

    # Donchian breakout
    donchian_high = high.rolling(20, min_periods=20).max()
    out["donchian_20_breakout"] = (close >= donchian_high.shift(1)).astype(float)

    # Relative strength vs index
    stock_ret_60 = close.pct_change(60)
    out["stock_ret_60"] = stock_ret_60
    if index_ohlcv is not None and not index_ohlcv.empty:
        idx_close = pd.to_numeric(index_ohlcv["close"], errors="coerce")
        idx_dates = pd.to_datetime(index_ohlcv["trade_date"]).dt.normalize()
        stock_dates = pd.to_datetime(df["trade_date"]).dt.normalize()
        idx_ret_map = dict(zip(idx_dates, idx_close.pct_change(60)))
        out["rs_60"] = pd.Series(
            [stock_ret_60.iloc[i] - idx_ret_map.get(d, 0.0) for i, d in enumerate(stock_dates)],
            index=df.index,
            dtype=np.float64,
        )
    else:
        out["rs_60"] = stock_ret_60

    return out


FEATURE_COLUMNS = [
    "ma20_slope",
    "ma60_slope",
    "close_above_ma20",
    "close_above_ma60",
    "atr_pct",
    "atr_pct_rank",
    "volume_ratio_20",
    "donchian_20_breakout",
    "stock_ret_60",
    "rs_60",
]


def build_training_samples(
    ohlcv: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    *,
    index_ohlcv: pd.DataFrame | None = None,
    forward_return_days: int = 20,
    label_type: str = "profit",
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build (X, y, dates) from BUY signal dates.

    Parameters
    ----------
    label_type:
        - ``"profit"``: forward *forward_return_days* close return > 0
        - ``"risk_reward"``: max forward 20d return > |min forward 10d return|
    """
    features_df = build_signal_features(ohlcv, index_ohlcv=index_ohlcv)
    close = pd.to_numeric(ohlcv["close"], errors="coerce")

    # Build date → index mapping
    trade_dates = pd.to_datetime(ohlcv["trade_date"]).dt.normalize()
    date_to_pos = {d.date(): i for i, d in enumerate(trade_dates)}

    X_rows = []
    y_rows = []
    valid_dates = []

    for sig_date in signal_dates:
        sd = sig_date.date()
        if sd not in date_to_pos:
            continue
        pos = date_to_pos[sd]
        if pos >= len(close) - forward_return_days:
            continue

        # Check feature availability
        feat_row = features_df.iloc[pos]
        if feat_row[FEATURE_COLUMNS].isna().all():
            continue

        # Build label
        if label_type == "profit":
            fwd_ret = (close.iloc[pos + forward_return_days] / close.iloc[pos]) - 1.0
            label = 1.0 if fwd_ret > 0 else 0.0
        elif label_type == "risk_reward":
            fwd_prices = close.iloc[pos : pos + forward_return_days + 1]
            fwd_rets = (fwd_prices / close.iloc[pos]) - 1.0
            max_gain = fwd_rets.max()
            max_loss = fwd_rets.iloc[: min(10, len(fwd_rets))].min()
            label = 1.0 if max_gain > abs(max_loss) else 0.0
        else:
            raise ValueError(f"unknown label_type: {label_type}")

        X_rows.append([feat_row[c] for c in FEATURE_COLUMNS])
        y_rows.append(label)
        valid_dates.append(sig_date)

    X = np.array(X_rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0)
    y = np.array(y_rows, dtype=np.float64)
    return X, y, pd.DatetimeIndex(valid_dates)


class LogisticMetaModel:
    """L2-regularised logistic regression for meta-labeling.

    Trained via batch gradient descent with ridge penalty.
    """

    def __init__(self, l2_penalty: float = 0.1, learning_rate: float = 0.01,
                 max_iter: int = 2000, tol: float = 1e-6):
        self.l2 = l2_penalty
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.feature_names_: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: list[str] | None = None) -> LogisticMetaModel:
        n, d = X.shape
        self.feature_names_ = feature_names or [f"f{i}" for i in range(d)]
        # Standardise features
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0)
        self.x_std_[self.x_std_ == 0] = 1.0
        Xs = (X - self.x_mean_) / self.x_std_

        w = np.zeros(d, dtype=np.float64)
        b = 0.0
        prev_loss = np.inf

        for it in range(self.max_iter):
            logits = Xs @ w + b
            p = _sigmoid(logits)
            err = p - y
            grad_w = (Xs.T @ err) / n + self.l2 * w
            grad_b = err.mean()
            w -= self.lr * grad_w
            b -= self.lr * grad_b

            loss = -np.mean(y * np.log(np.clip(p, 1e-15, 1.0))
                            + (1 - y) * np.log(np.clip(1 - p, 1e-15, 1.0)))
            loss += 0.5 * self.l2 * (w @ w)
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        self.coef_ = w
        self.intercept_ = b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model not fitted")
        Xs = (X - self.x_mean_) / self.x_std_
        return _sigmoid(Xs @ self.coef_ + self.intercept_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.float64)

    def get_feature_importance(self) -> dict[str, float]:
        if self.coef_ is None:
            return {}
        return dict(zip(self.feature_names_, self.coef_.tolist()))


def run_meta_label_wfo(
    ohlcv: pd.DataFrame,
    signal_dates_all: pd.DatetimeIndex,
    *,
    index_ohlcv: pd.DataFrame | None = None,
    train_days: int = 504,
    oos_days: int = 126,
    l2_penalty: float = 0.1,
    label_type: str = "profit",
    min_signals_per_fold: int = 5,
) -> list[MetaLabelResult]:
    """Walk-forward evaluation of meta-label model.

    For each fold:
    1. Build training samples from train-period signal dates
    2. Train logistic regression
    3. Predict on OOS-period signal dates
    4. Compare predicted vs actual labels
    """
    ohlcv = ohlcv.copy()
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"]).dt.normalize()
    ohlcv = ohlcv.sort_values("trade_date").reset_index(drop=True)

    trade_dates = ohlcv["trade_date"]
    results: list[MetaLabelResult] = []

    start = 0
    fold = 0
    n_total = len(ohlcv)
    while start + train_days + oos_days < n_total:
        train_end = start + train_days
        oos_end = min(train_end + oos_days, n_total)

        train_mask = (signal_dates_all >= trade_dates.iloc[start]) & (signal_dates_all <= trade_dates.iloc[train_end - 1])
        test_end_idx = min(oos_end, n_total) - 1
        test_mask = (signal_dates_all > trade_dates.iloc[train_end - 1]) & (signal_dates_all <= trade_dates.iloc[test_end_idx])

        train_dates = signal_dates_all[train_mask]
        test_dates = signal_dates_all[test_mask]

        train_df = ohlcv.iloc[start:train_end]
        test_df = ohlcv.iloc[train_end:oos_end]

        r = MetaLabelResult(fold=fold, train_n=len(train_dates), test_n=len(test_dates))

        if len(train_dates) < min_signals_per_fold or len(test_dates) < min_signals_per_fold:
            results.append(r)
            fold += 1
            start += oos_days
            continue

        X_train, y_train, _ = build_training_samples(
            train_df, train_dates, index_ohlcv=index_ohlcv,
            label_type=label_type,
        )
        X_test, y_test, _ = build_training_samples(
            test_df, test_dates, index_ohlcv=index_ohlcv,
            label_type=label_type,
        )

        if len(X_train) < min_signals_per_fold or len(X_test) < min_signals_per_fold:
            results.append(r)
            fold += 1
            start += oos_days
            continue

        model = LogisticMetaModel(l2_penalty=l2_penalty)
        model.fit(X_train, y_train, feature_names=FEATURE_COLUMNS)

        r.coef = model.coef_.tolist() if model.coef_ is not None else []
        r.intercept = model.intercept_
        r.feature_importance = model.get_feature_importance()

        # Train-set metrics
        train_pred = model.predict(X_train)
        r.train_precision = float(np.mean(train_pred == y_train))

        # Test-set metrics
        test_prob = model.predict_proba(X_test)
        test_pred = model.predict(X_test)
        r.test_precision = float(np.mean(test_pred == y_test))
        r.test_n_pred_positive = int(test_pred.sum())
        tp = int(((test_pred == 1) & (y_test == 1)).sum())
        fn = int(((test_pred == 0) & (y_test == 1)).sum())
        r.test_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        results.append(r)
        fold += 1
        start += oos_days

    return results
