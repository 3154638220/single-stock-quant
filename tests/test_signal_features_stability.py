"""Stability checks for meta-label feature engineering."""

import numpy as np
import pandas as pd

from src.models.meta_label import FEATURE_COLUMNS, LogisticMetaModel, build_signal_features


def _feature_df(n: int = 520) -> pd.DataFrame:
    x = np.arange(n, dtype=np.float64)
    close = 100.0 + 0.04 * x + 6.0 * np.sin(x / 17.0)
    volume = 1000.0 + 150.0 * np.cos(x / 11.0) + (x % 30) * 3.0
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2022-01-01", periods=n),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
            "turnover_rate": 1.0 + (x % 80) / 100.0,
        }
    )


def test_feature_importance_stability_across_folds():
    df = _feature_df()
    index_df = _feature_df()
    index_df["close"] = np.linspace(100.0, 118.0, len(index_df))

    feats = build_signal_features(df, index_ohlcv=index_df).replace([np.inf, -np.inf], np.nan)
    sample = feats[FEATURE_COLUMNS].iloc[260::4].dropna(how="all").fillna(0.0)
    score = sample["pos_52w"] + 0.8 * sample["stock_trend_quality"]
    labels = (score > score.median()).astype(np.float64).to_numpy()

    top_features: list[str] = []
    fold_size = 40
    for start in range(0, len(sample) - fold_size + 1, 20):
        X = sample.iloc[start : start + fold_size].to_numpy(dtype=np.float64)
        y = labels[start : start + fold_size]
        if len(np.unique(y)) < 2:
            continue
        model = LogisticMetaModel(l2_penalty=0.01, max_iter=3000)
        model.fit(X, y, feature_names=FEATURE_COLUMNS)
        ranked = sorted(
            model.get_feature_importance().items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        top_features.extend(name for name, _ in ranked[:3])

    counts = {name: top_features.count(name) for name in set(top_features)}
    repeated = [name for name, count in counts.items() if count >= 2]
    assert {"pos_52w", "stock_trend_quality"} & set(repeated)
    assert len(repeated) >= 2
