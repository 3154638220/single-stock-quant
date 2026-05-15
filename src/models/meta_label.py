"""Meta-labeling: predict whether a BUY signal is worth executing.

Lightweight logistic-regression model trained on per-signal features.
All evaluation is WFO-based — no random train/test split.

Stage 23 improvements:
- New ``label_type="profit_aware"``: requires both positive forward return AND
  max drawdown during holding period < threshold
- Signal-context features: DK trend state, run length, entry quality
- Symbol-level recent performance meta-features
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.features.sector_features import compute_industry_rs_20


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


def _rolling_last_rank(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Percentile rank of the current value within its trailing window."""
    return s.rolling(window, min_periods=min_periods).apply(
        lambda x: (x.iloc[-1] >= x).mean(),
        raw=False,
    )


def _rolling_beta(stock_ret: pd.Series, index_ret: pd.Series, window: int = 120) -> pd.Series:
    """Rolling beta of stock returns versus benchmark returns."""
    cov = stock_ret.rolling(window, min_periods=max(30, window // 2)).cov(index_ret)
    var = index_ret.rolling(window, min_periods=max(30, window // 2)).var()
    return cov / var.replace(0.0, np.nan)


def build_signal_features(
    ohlcv: pd.DataFrame,
    *,
    index_ohlcv: pd.DataFrame | None = None,
    dk_trend_state: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build feature matrix for every row in *ohlcv*.

    Returns a DataFrame with feature columns indexed by position.

    When *dk_trend_state* is provided (columns: trade_date, dk_color,
    dk_run_len), signal-context features such as trend run length and
    freshness are appended.
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
    # ATR expansion: short-term vol / long-term vol
    atr60 = tr.rolling(60, min_periods=60).mean()
    out["atr_expansion"] = (atr / atr60).fillna(1.0)

    # Volume ratio
    avg_vol_20 = volume.rolling(20, min_periods=20).mean()
    out["volume_ratio_20"] = volume / avg_vol_20

    # Donchian breakout
    donchian_high = high.rolling(20, min_periods=20).max()
    out["donchian_20_breakout"] = (close >= donchian_high.shift(1)).astype(float)

    # Relative strength vs index
    stock_ret_60 = close.pct_change(60)
    out["stock_ret_60"] = stock_ret_60
    stock_daily_ret = close.pct_change(fill_method=None)
    index_daily_ret = pd.Series(0.0, index=df.index, dtype=np.float64)
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
        idx_daily_ret_map = dict(zip(idx_dates, idx_close.pct_change(fill_method=None)))
        index_daily_ret = pd.Series(
            [idx_daily_ret_map.get(d, 0.0) for d in stock_dates],
            index=df.index,
            dtype=np.float64,
        )
    else:
        out["rs_60"] = stock_ret_60

    # S4 — industry/sector 20-day relative strength
    out["industry_rs_20"] = compute_industry_rs_20(
        df, index_ohlcv=index_ohlcv,
    )

    # Phase 11 feature set: longer-horizon price location, price/volume
    # behaviour, trend consistency, acceleration, and benchmark sensitivity.
    high_252 = high.rolling(252, min_periods=120).max()
    low_252 = low.rolling(252, min_periods=120).min()
    out["pos_52w"] = (close - low_252) / (high_252 - low_252).replace(0.0, np.nan)

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd_line - macd_signal
    out["macd_hist_dir"] = (macd_hist.diff() > 0).rolling(3, min_periods=3).sum().eq(3).astype(float)

    turnover_proxy = pd.to_numeric(df.get("turnover_rate", volume * close), errors="coerce")
    out["turnover_rank_60"] = _rolling_last_rank(turnover_proxy, 60, 30)

    out["beta_120"] = _rolling_beta(stock_daily_ret, index_daily_ret, window=120)

    ma10 = close.rolling(10, min_periods=10).mean()
    out["close_accel_10"] = ma10.diff().diff()

    out["vol_price_corr_20"] = volume.rolling(20, min_periods=10).corr(stock_daily_ret)

    # ── Stage 23: signal-context features ──
    # Entry quality: how far is close from MA20 in ATR terms
    out["entry_distance_ma20_atr"] = (close - ma20) / atr.replace(0.0, np.nan)

    # Overextension: close relative to 20-day high-low range
    high20 = high.rolling(20, min_periods=20).max()
    low20 = low.rolling(20, min_periods=20).min()
    out["close_in_20d_range"] = (close - low20) / (high20 - low20).replace(0.0, np.nan)

    # Trend alignment: MA20 slope direction * MA60 slope direction
    ma20_dir = (ma20.diff() > 0).astype(float) * 2 - 1
    ma60_dir = (ma60.diff() > 0).astype(float) * 2 - 1
    out["trend_alignment"] = ma20_dir * ma60_dir  # +1 both up, -1 both down, 0 mixed

    # Consecutive up days
    close_dir = (close.diff() > 0).astype(float)
    out["consecutive_up_5"] = close_dir.rolling(5, min_periods=5).sum() / 5.0

    # Gap from recent high (pullback depth)
    high10 = high.rolling(10, min_periods=10).max()
    out["pullback_from_high10"] = (close - high10) / high10.replace(0.0, np.nan) * 100

    # Volume trend: is volume expanding vs 20 days ago
    out["volume_trend"] = (volume.rolling(5, min_periods=5).mean() /
                           volume.rolling(20, min_periods=20).mean()).fillna(1.0)

    # Holding regime from index (bull=1, bear=-1, ranging=0)
    if index_ohlcv is not None and not index_ohlcv.empty:
        idx_close_reg = pd.to_numeric(index_ohlcv["close"], errors="coerce")
        idx_ret60_reg = idx_close_reg.pct_change(60)
        idx_dates_reg = pd.to_datetime(index_ohlcv["trade_date"]).dt.normalize()
        regime_map = {}
        for _j, _d in enumerate(idx_dates_reg):
            _r = idx_ret60_reg.iloc[_j]
            if pd.notna(_r):
                regime_map[_d] = 1.0 if _r > 0.10 else (-1.0 if _r < -0.10 else 0.0)
        stock_dates_reg = pd.to_datetime(df["trade_date"]).dt.normalize()
        out["holding_regime"] = pd.Series(
            [regime_map.get(d, 0.0) for d in stock_dates_reg],
            index=df.index,
            dtype=np.float64,
        )
    else:
        out["holding_regime"] = 0.0

    # DK trend context (if provided)
    if dk_trend_state is not None:
        dk = dk_trend_state.copy()
        dk["trade_date"] = pd.to_datetime(dk["trade_date"]).dt.normalize()
        stock_dates = pd.to_datetime(df["trade_date"]).dt.normalize()

        # Basic DK context
        dk_map = dict(zip(dk["trade_date"], dk["dk_run_len"]))
        out["dk_run_len"] = pd.Series(
            [dk_map.get(d, -1) for d in stock_dates],
            index=df.index,
            dtype=np.float64,
        )
        dk_color_map = dict(zip(dk["trade_date"], dk["dk_color"]))
        out["dk_is_red"] = pd.Series(
            [1.0 if str(dk_color_map.get(d, "")).lower() == "red" else 0.0 for d in stock_dates],
            index=df.index,
            dtype=np.float64,
        )

        # S4 — new structural DK features
        dk_value_map = dict(zip(dk["trade_date"], pd.to_numeric(dk.get("dk_value", pd.Series(0.0)), errors="coerce")))
        dk_color_str_map = dict(zip(dk["trade_date"], dk.get("dk_color", pd.Series(""))))

        # dk_value_pct_rank: percentile rank over 120 days
        dk_vals_aligned = pd.Series(
            [dk_value_map.get(d, 0.0) for d in stock_dates],
            index=df.index,
            dtype=np.float64,
        )
        out["dk_value_pct_rank"] = (
            dk_vals_aligned.rolling(120, min_periods=60)
            .apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)
        )

        # stock_trend_quality: |mean(dk_value)| / std(dk_value) over 20 days
        dk_roll_mean = dk_vals_aligned.rolling(20, min_periods=10).mean()
        dk_roll_std = dk_vals_aligned.rolling(20, min_periods=10).std()
        out["stock_trend_quality"] = (
            dk_roll_mean.abs() / dk_roll_std.replace(0.0, np.nan)
        ).fillna(0.0)

        # signal_recency: days since last DK color flip (red↔green)
        colors_aligned = [
            str(dk_color_str_map.get(d, "")).lower() for d in stock_dates
        ]
        recency = np.full(len(colors_aligned), -1.0, dtype=np.float64)
        last_flip = -1
        for _j in range(len(colors_aligned)):
            if _j > 0 and colors_aligned[_j] != colors_aligned[_j - 1] and colors_aligned[_j] in ("red", "green"):
                last_flip = _j
            if last_flip >= 0:
                recency[_j] = float(_j - last_flip)
        out["signal_recency"] = recency
    else:
        out["dk_run_len"] = -1.0
        out["dk_is_red"] = 0.0
        out["dk_value_pct_rank"] = 0.5
        out["stock_trend_quality"] = 0.0
        out["signal_recency"] = -1.0

    return out


FEATURE_COLUMNS = [
    "ma20_slope",
    "ma60_slope",
    "close_above_ma60",
    "atr_pct",
    "atr_pct_rank",
    "atr_expansion",
    "volume_ratio_20",
    "donchian_20_breakout",
    "stock_ret_60",
    "rs_60",
    "industry_rs_20",
    "pos_52w",
    "macd_hist_dir",
    "turnover_rank_60",
    "beta_120",
    "close_accel_10",
    "vol_price_corr_20",
    "entry_distance_ma20_atr",
    "close_in_20d_range",
    "trend_alignment",
    "consecutive_up_5",
    "pullback_from_high10",
    "volume_trend",
    "dk_run_len",
    "dk_is_red",
    "stock_trend_quality",
    "holding_regime",
    "signal_recency",
    "dk_value_pct_rank",
]


def build_training_samples(
    ohlcv: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    *,
    index_ohlcv: pd.DataFrame | None = None,
    forward_return_days: int = 20,
    label_type: str = "profit_aware",
    max_drawdown_threshold: float = 0.08,
    dk_trend_state: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build (X, y, dates) from BUY signal dates.

    Parameters
    ----------
    label_type:
        - ``"profit"``: forward *forward_return_days* close return > 0
        - ``"risk_reward"``: max forward 20d return > |min forward 10d return|
        - ``"profit_aware"``: forward return > 0 AND max drawdown during
          holding period < *max_drawdown_threshold*
        - ``"label_v1"``: 10-day forward return > 0
        - ``"label_v2"``: 10-day forward return > ATR(14)
        - ``"label_v3"``: MFE/MAE > 2.0 (profit factor over 10 days)
        - ``"label_v4"``: 10-day return > 0 AND 5-day max DD < 2 * ATR(14)
    max_drawdown_threshold:
        Used for ``"profit_aware"`` label type.
    """
    features_df = build_signal_features(ohlcv, index_ohlcv=index_ohlcv, dk_trend_state=dk_trend_state)
    close = pd.to_numeric(ohlcv["close"], errors="coerce")
    open_price = pd.to_numeric(ohlcv.get("open", close), errors="coerce")
    # Pre-compute ATR(14) as a decimal fraction for new label types
    high_s = pd.to_numeric(ohlcv["high"], errors="coerce")
    low_s = pd.to_numeric(ohlcv["low"], errors="coerce")
    prev_close_s = close.shift(1)
    tr_s = pd.concat(
        [high_s - low_s, (high_s - prev_close_s).abs(), (low_s - prev_close_s).abs()], axis=1
    ).max(axis=1)
    atr_decimal = (tr_s.rolling(14, min_periods=14).mean() / close).fillna(0.01)

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
        elif label_type == "profit_aware":
            # Use open-to-open path for realistic execution
            entry_price = open_price.iloc[pos]
            fwd_path = open_price.iloc[pos + 1 : pos + forward_return_days + 1]
            if len(fwd_path) == 0 or entry_price <= 0:
                continue
            fwd_rets = fwd_path.to_numpy(dtype=np.float64) / entry_price - 1.0
            max_dd = float(np.min(fwd_rets))  # worst point during holding
            fwd_ret = float(close.iloc[pos + forward_return_days] / close.iloc[pos] - 1.0)
            # Must: (a) end positive, (b) never drawdown beyond threshold
            label = 1.0 if (fwd_ret > 0 and max_dd > -max_drawdown_threshold) else 0.0
        elif label_type == "label_v1":
            fwd_ret = (close.iloc[pos + 10] / close.iloc[pos]) - 1.0 if pos + 10 < len(close) else -1.0
            label = 1.0 if fwd_ret > 0 else 0.0
        elif label_type == "label_v2":
            atr14 = float(atr_decimal.iloc[pos]) if pos < len(atr_decimal) and np.isfinite(atr_decimal.iloc[pos]) else 0.01
            if atr14 <= 0:
                atr14 = 0.01
            fwd_ret = (close.iloc[pos + 10] / close.iloc[pos]) - 1.0 if pos + 10 < len(close) else -1.0
            label = 1.0 if fwd_ret > atr14 else 0.0
        elif label_type == "label_v3":
            fwd_prices = close.iloc[pos : min(pos + 11, len(close))]
            if len(fwd_prices) < 2:
                continue
            fwd_rets = (fwd_prices / close.iloc[pos]) - 1.0
            mfe = float(fwd_rets.max())
            mae = float(fwd_rets.min())
            label = 1.0 if (mae < 0 and mfe > 0 and abs(mfe / mae) > 2.0) else 0.0
        elif label_type == "label_v4":
            atr14 = float(atr_decimal.iloc[pos]) if pos < len(atr_decimal) and np.isfinite(atr_decimal.iloc[pos]) else 0.01
            if atr14 <= 0:
                atr14 = 0.01
            fwd_ret = (close.iloc[pos + 10] / close.iloc[pos]) - 1.0 if pos + 10 < len(close) else -1.0
            fwd_5 = close.iloc[pos : min(pos + 6, len(close))]
            if len(fwd_5) < 2:
                continue
            fwd_5_rets = (fwd_5 / close.iloc[pos]) - 1.0
            max_dd_5 = float(fwd_5_rets.min())
            label = 1.0 if (fwd_ret > 0 and max_dd_5 > -2.0 * atr14) else 0.0
        else:
            raise ValueError(f"unknown label_type: {label_type}")

        X_rows.append([feat_row.get(c, 0.0) for c in FEATURE_COLUMNS])
        y_rows.append(label)
        valid_dates.append(sig_date)

    X = np.array(X_rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y_rows, dtype=np.float64)
    return X, y, pd.DatetimeIndex(valid_dates)


def build_daily_labels(
    ohlcv: pd.DataFrame,
    dk_trend: pd.DataFrame,
    *,
    index_ohlcv: pd.DataFrame | None = None,
    forward_days: int = 10,
    label_type: str = "label_v4",
    dk_trend_state: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build (X, y, dates) from ALL DK red days, not just signal days.

    This expands training samples from ~50 (signal-level) to ~500+ (daily-level)
    per stock, giving the meta-model enough data to learn meaningful patterns.

    Only days with ``dk_color == "red"`` are included, since those are the days
    the strategy would consider being in a position.

    Parameters
    ----------
    forward_days:
        Number of days forward for return calculation (default 10).
    label_type:
        Same label types as ``build_training_samples``. Default ``"label_v4"``:
        10-day return > 0 AND 5-day max DD < 2 * ATR(14).
    """
    features_df = build_signal_features(ohlcv, index_ohlcv=index_ohlcv, dk_trend_state=dk_trend_state)
    close = pd.to_numeric(ohlcv["close"], errors="coerce")
    high_s = pd.to_numeric(ohlcv["high"], errors="coerce")
    low_s = pd.to_numeric(ohlcv["low"], errors="coerce")
    prev_close_s = close.shift(1)
    tr_s = pd.concat(
        [high_s - low_s, (high_s - prev_close_s).abs(), (low_s - prev_close_s).abs()], axis=1
    ).max(axis=1)
    atr_decimal = (tr_s.rolling(14, min_periods=14).mean() / close).fillna(0.01)

    # Align dk_trend with ohlcv by date
    trend_dates = pd.to_datetime(dk_trend["trade_date"]).dt.normalize()
    ohlcv_dates = pd.to_datetime(ohlcv["trade_date"]).dt.normalize()
    date_to_ohlcv_pos = {d.date(): i for i, d in enumerate(ohlcv_dates)}

    X_rows = []
    y_rows = []
    valid_dates = []

    for ti, trend_row in dk_trend.iterrows():
        if str(trend_row.get("dk_color", "")) != "red":
            continue
        td = trend_dates[ti].date()
        if td not in date_to_ohlcv_pos:
            continue
        pos = date_to_ohlcv_pos[td]
        if pos >= len(close) - forward_days:
            continue

        feat_row = features_df.iloc[pos]
        if feat_row[FEATURE_COLUMNS].isna().all():
            continue

        # Build label (mirrors build_training_samples logic)
        if label_type == "profit":
            fwd_ret = (close.iloc[pos + forward_days] / close.iloc[pos]) - 1.0
            label = 1.0 if fwd_ret > 0 else 0.0
        elif label_type == "label_v1":
            fwd_ret = (close.iloc[pos + 10] / close.iloc[pos]) - 1.0 if pos + 10 < len(close) else -1.0
            label = 1.0 if fwd_ret > 0 else 0.0
        elif label_type == "label_v2":
            atr14 = float(atr_decimal.iloc[pos]) if pos < len(atr_decimal) and np.isfinite(atr_decimal.iloc[pos]) else 0.01
            if atr14 <= 0:
                atr14 = 0.01
            fwd_ret = (close.iloc[pos + 10] / close.iloc[pos]) - 1.0 if pos + 10 < len(close) else -1.0
            label = 1.0 if fwd_ret > atr14 else 0.0
        elif label_type == "label_v3":
            fwd_prices = close.iloc[pos : min(pos + forward_days + 1, len(close))]
            if len(fwd_prices) < 2:
                continue
            fwd_rets = (fwd_prices / close.iloc[pos]) - 1.0
            mfe = float(fwd_rets.max())
            mae = float(fwd_rets.min())
            label = 1.0 if (mae < 0 and mfe > 0 and abs(mfe / mae) > 2.0) else 0.0
        elif label_type == "label_v4":
            atr14 = float(atr_decimal.iloc[pos]) if pos < len(atr_decimal) and np.isfinite(atr_decimal.iloc[pos]) else 0.01
            if atr14 <= 0:
                atr14 = 0.01
            fwd_ret = (close.iloc[pos + 10] / close.iloc[pos]) - 1.0 if pos + 10 < len(close) else -1.0
            fwd_5 = close.iloc[pos : min(pos + 6, len(close))]
            if len(fwd_5) < 2:
                continue
            fwd_5_rets = (fwd_5 / close.iloc[pos]) - 1.0
            max_dd_5 = float(fwd_5_rets.min())
            label = 1.0 if (fwd_ret > 0 and max_dd_5 > -2.0 * atr14) else 0.0
        else:
            raise ValueError(f"unknown label_type: {label_type}")

        X_rows.append([feat_row.get(c, 0.0) for c in FEATURE_COLUMNS])
        y_rows.append(label)
        valid_dates.append(pd.Timestamp(td))

    X = np.array(X_rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
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
    label_type: str = "profit_aware",
    max_drawdown_threshold: float = 0.08,
    min_signals_per_fold: int = 5,
    dk_trend_state: pd.DataFrame | None = None,
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
            label_type=label_type, max_drawdown_threshold=max_drawdown_threshold,
            dk_trend_state=dk_trend_state,
        )
        X_test, y_test, _ = build_training_samples(
            test_df, test_dates, index_ohlcv=index_ohlcv,
            label_type=label_type, max_drawdown_threshold=max_drawdown_threshold,
            dk_trend_state=dk_trend_state,
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
