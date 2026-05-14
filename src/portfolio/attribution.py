"""Attribution helpers for portfolio ranking scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_score_forward_return_attribution(
    daily_long: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 5, 20),
    n_quantiles: int = 5,
    min_score: float = 0.0,
    date_col: str = "trade_date",
    sym_col: str = "symbol",
) -> pd.DataFrame:
    """Bucket ranking scores and measure subsequent open-to-open returns.

    A score observed on day ``t`` is assumed to enter at the next open, matching
    the portfolio engine's T+1 open execution convention.
    """
    if daily_long.empty:
        raise ValueError("daily_long is empty")
    if scores.empty:
        raise ValueError("scores is empty")
    missing = {date_col, sym_col, "open"} - set(daily_long.columns)
    if missing:
        raise ValueError(f"daily_long missing columns: {sorted(missing)}")

    df = daily_long.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df[sym_col] = df[sym_col].astype(str).str.zfill(6)
    open_wide = df.pivot(index=date_col, columns=sym_col, values="open").sort_index().astype(np.float64)

    score_wide = scores.copy()
    score_wide.index = pd.to_datetime(score_wide.index).normalize()
    score_wide.columns = score_wide.columns.astype(str).str.zfill(6)
    score_wide = score_wide.reindex(index=open_wide.index, columns=open_wide.columns)

    rows: list[dict[str, float | int | str]] = []
    for horizon in horizons:
        h = max(int(horizon), 1)
        fwd = open_wide.shift(-(h + 1)) / open_wide.shift(-1) - 1.0
        panel = (
            _stack_wide(score_wide)
            .rename("score")
            .to_frame()
            .join(_stack_wide(fwd).rename("forward_return"))
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["score", "forward_return"])
        )
        panel = panel[panel["score"] > float(min_score)]
        if panel.empty:
            continue

        bucketed = _assign_score_quantiles(panel, n_quantiles=n_quantiles)
        for bucket, g in bucketed.groupby("bucket", observed=True):
            returns = g["forward_return"].astype(np.float64)
            scores_bucket = g["score"].astype(np.float64)
            rows.append(
                {
                    "horizon": h,
                    "bucket": str(bucket),
                    "n": int(len(g)),
                    "score_min": float(scores_bucket.min()),
                    "score_mean": float(scores_bucket.mean()),
                    "score_max": float(scores_bucket.max()),
                    "mean_forward_return": float(returns.mean()),
                    "median_forward_return": float(returns.median()),
                    "win_rate": float((returns > 0).mean()),
                    "p10_forward_return": float(returns.quantile(0.10)),
                    "p90_forward_return": float(returns.quantile(0.90)),
                }
            )

    return pd.DataFrame(rows)


def summarize_score_monotonicity(bucket_attribution: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether higher score buckets earn higher future returns."""
    if bucket_attribution.empty:
        return pd.DataFrame(
            columns=["horizon", "top_minus_bottom", "bucket_return_corr", "is_monotonic"]
        )

    rows: list[dict[str, float | int | bool]] = []
    for horizon, g in bucket_attribution.groupby("horizon", sort=True):
        ordered = g.copy()
        ordered["_bucket_num"] = ordered["bucket"].astype(str).str.extract(r"(\d+)").astype(int)
        ordered = ordered.sort_values("_bucket_num")
        mean_ret = ordered["mean_forward_return"].astype(np.float64)
        bucket_num = ordered["_bucket_num"].astype(np.float64)
        corr = float(bucket_num.corr(mean_ret)) if len(ordered) >= 2 else float("nan")
        top_minus_bottom = (
            float(mean_ret.iloc[-1] - mean_ret.iloc[0]) if len(ordered) >= 2 else float("nan")
        )
        rows.append(
            {
                "horizon": int(horizon),
                "top_minus_bottom": top_minus_bottom,
                "bucket_return_corr": corr,
                "is_monotonic": bool(mean_ret.is_monotonic_increasing),
            }
        )
    return pd.DataFrame(rows)


def compute_candidate_forward_return_breakdown(
    daily_long: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 5, 20),
    min_score: float = 0.0,
    index_ohlcv: pd.DataFrame | None = None,
    industry_map: dict[str, str] | None = None,
    group_by: tuple[str, ...] = ("symbol",),
    date_col: str = "trade_date",
    sym_col: str = "symbol",
) -> pd.DataFrame:
    """Measure candidate forward returns by symbol, regime, or industry.

    ``scores`` is treated as the candidate panel: rows with ``score > min_score``
    are analyzed using the same T+1 open execution convention as the portfolio
    ranking attribution.
    """
    panel = _score_forward_return_panel(
        daily_long,
        scores,
        horizons=horizons,
        min_score=min_score,
        date_col=date_col,
        sym_col=sym_col,
    )
    if panel.empty:
        return pd.DataFrame(
            columns=[
                "horizon",
                *group_by,
                "n",
                "share_of_candidates",
                "score_mean",
                "mean_forward_return",
                "median_forward_return",
                "win_rate",
                "p10_forward_return",
                "p90_forward_return",
                "return_contribution",
            ]
        )

    regime_source = index_ohlcv
    if regime_source is None or regime_source.empty:
        regime_source = _cross_section_regime_source(daily_long, date_col=date_col, sym_col=sym_col)
    panel["market_regime"] = _market_regime_by_date(regime_source, panel[date_col], date_col=date_col)
    normalized_industry = {
        str(k).zfill(6): str(v) for k, v in (industry_map or {}).items()
    }
    panel["industry"] = panel[sym_col].map(normalized_industry).fillna("unknown")

    valid_groupers = {"symbol", "market_regime", "industry"}
    groupers = tuple(dict.fromkeys(str(g).strip() for g in group_by if str(g).strip()))
    unknown = set(groupers) - valid_groupers
    if unknown:
        raise ValueError(f"unknown group_by values: {sorted(unknown)}")
    if not groupers:
        groupers = ("symbol",)

    total_by_horizon = panel.groupby("horizon", observed=True).size()
    grouped = panel.groupby(["horizon", *groupers], observed=True, dropna=False)
    out = grouped.agg(
        n=("forward_return", "size"),
        score_mean=("score", "mean"),
        mean_forward_return=("forward_return", "mean"),
        median_forward_return=("forward_return", "median"),
        win_rate=("forward_return", lambda x: float((x > 0).mean())),
        p10_forward_return=("forward_return", lambda x: float(x.quantile(0.10))),
        p90_forward_return=("forward_return", lambda x: float(x.quantile(0.90))),
    ).reset_index()
    out["share_of_candidates"] = out["n"] / out["horizon"].map(total_by_horizon).astype(np.float64)
    out["return_contribution"] = out["share_of_candidates"] * out["mean_forward_return"]
    ordered_cols = [
        "horizon",
        *groupers,
        "n",
        "share_of_candidates",
        "score_mean",
        "mean_forward_return",
        "median_forward_return",
        "win_rate",
        "p10_forward_return",
        "p90_forward_return",
        "return_contribution",
    ]
    return out[ordered_cols].sort_values(
        ["horizon", "mean_forward_return", "n"],
        ascending=[True, True, False],
        ignore_index=True,
    )


def calibrate_scores_by_forward_returns(
    daily_long: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    horizon: int = 5,
    lookback_days: int = 252,
    n_quantiles: int = 5,
    min_observations: int = 100,
    calibration_strength: float = 0.70,
    min_score: float = 0.0,
    date_col: str = "trade_date",
    sym_col: str = "symbol",
) -> pd.DataFrame:
    """Calibrate scores with rolling historical bucket forward returns.

    For a score observed on date ``t``, only score buckets whose forward returns
    are fully known by ``t`` are used. With a 5-day horizon, the newest training
    score date is therefore ``t-6`` because the portfolio enters at ``t+1`` open.
    """
    if daily_long.empty:
        raise ValueError("daily_long is empty")
    if scores.empty:
        raise ValueError("scores is empty")
    missing = {date_col, sym_col, "open"} - set(daily_long.columns)
    if missing:
        raise ValueError(f"daily_long missing columns: {sorted(missing)}")

    h = max(int(horizon), 1)
    lookback = max(int(lookback_days), h + 2)
    q = max(int(n_quantiles), 2)
    min_obs = max(int(min_observations), q)
    strength = float(np.clip(calibration_strength, 0.0, 1.0))

    df = daily_long.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df[sym_col] = df[sym_col].astype(str).str.zfill(6)
    open_wide = df.pivot(index=date_col, columns=sym_col, values="open").sort_index().astype(np.float64)

    score_wide = scores.copy().astype(np.float64)
    score_wide.index = pd.to_datetime(score_wide.index).normalize()
    score_wide.columns = score_wide.columns.astype(str).str.zfill(6)
    score_wide = score_wide.reindex(index=open_wide.index, columns=open_wide.columns)
    forward_returns = open_wide.shift(-(h + 1)) / open_wide.shift(-1) - 1.0

    calibrated = score_wide.copy()
    for pos, dt in enumerate(score_wide.index):
        train_end = pos - h - 1
        if train_end < 0:
            continue
        train_start = max(0, train_end - lookback + 1)
        score_hist = score_wide.iloc[train_start : train_end + 1]
        ret_hist = forward_returns.iloc[train_start : train_end + 1]
        panel = (
            _stack_wide(score_hist)
            .rename("score")
            .to_frame()
            .join(_stack_wide(ret_hist).rename("forward_return"))
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["score", "forward_return"])
        )
        panel = panel[panel["score"] > float(min_score)]
        if len(panel) < min_obs:
            continue

        edges = _score_quantile_edges(panel["score"], n_quantiles=q)
        if len(edges) < 3:
            continue
        buckets = pd.cut(panel["score"], bins=edges, labels=False, include_lowest=True)
        bucket_means = panel.groupby(buckets, observed=True)["forward_return"].mean().dropna()
        if bucket_means.empty or bucket_means.nunique(dropna=True) <= 1:
            continue
        bucket_scores = bucket_means.rank(method="average", pct=True) * 100.0

        day = score_wide.loc[dt].astype(np.float64)
        active = day > float(min_score)
        if not bool(active.any()):
            continue
        day_buckets = pd.cut(day[active], bins=edges, labels=False, include_lowest=True)
        expected_score = day_buckets.map(bucket_scores).astype(np.float64)
        expected_score = expected_score.reindex(day[active].index).fillna(day[active])
        calibrated.loc[dt, active.index[active]] = (
            (1.0 - strength) * day[active] + strength * expected_score
        )

    return calibrated.clip(lower=0.0, upper=100.0)


def _score_forward_return_panel(
    daily_long: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    min_score: float,
    date_col: str,
    sym_col: str,
) -> pd.DataFrame:
    if daily_long.empty:
        raise ValueError("daily_long is empty")
    if scores.empty:
        raise ValueError("scores is empty")
    missing = {date_col, sym_col, "open"} - set(daily_long.columns)
    if missing:
        raise ValueError(f"daily_long missing columns: {sorted(missing)}")

    df = daily_long.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df[sym_col] = df[sym_col].astype(str).str.zfill(6)
    open_wide = df.pivot(index=date_col, columns=sym_col, values="open").sort_index().astype(np.float64)
    open_wide.index.name = date_col
    open_wide.columns.name = sym_col

    score_wide = scores.copy()
    score_wide.index = pd.to_datetime(score_wide.index).normalize()
    score_wide.columns = score_wide.columns.astype(str).str.zfill(6)
    score_wide = score_wide.reindex(index=open_wide.index, columns=open_wide.columns).astype(np.float64)
    score_wide.index.name = date_col
    score_wide.columns.name = sym_col

    rows = []
    for horizon in horizons:
        h = max(int(horizon), 1)
        fwd = open_wide.shift(-(h + 1)) / open_wide.shift(-1) - 1.0
        fwd.index.name = date_col
        fwd.columns.name = sym_col
        panel = (
            _stack_wide(score_wide)
            .rename("score")
            .to_frame()
            .join(_stack_wide(fwd).rename("forward_return"))
            .reset_index()
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["score", "forward_return"])
        )
        panel = panel[panel["score"] > float(min_score)]
        if panel.empty:
            continue
        panel["horizon"] = h
        rows.append(panel)

    if not rows:
        return pd.DataFrame(columns=[date_col, sym_col, "score", "forward_return", "horizon"])
    out = pd.concat(rows, ignore_index=True)
    out[sym_col] = out[sym_col].astype(str).str.zfill(6)
    return out


def _market_regime_by_date(
    index_ohlcv: pd.DataFrame | None,
    dates: pd.Series,
    *,
    date_col: str,
) -> pd.Series:
    if index_ohlcv is None or index_ohlcv.empty or date_col not in index_ohlcv or "close" not in index_ohlcv:
        return pd.Series("unknown", index=dates.index, dtype="object")

    idx = index_ohlcv.copy()
    idx[date_col] = pd.to_datetime(idx[date_col]).dt.normalize()
    close = pd.to_numeric(idx["close"], errors="coerce")
    by_date = pd.Series(close.to_numpy(dtype=np.float64), index=idx[date_col]).sort_index()
    ma60 = by_date.rolling(60, min_periods=60).mean()
    ret20 = by_date.pct_change(20, fill_method=None)
    regime = pd.Series("mixed", index=by_date.index, dtype="object")
    regime[(by_date > ma60) & (ret20 >= 0)] = "bull"
    regime[(by_date < ma60) & (ret20 < 0)] = "bear"
    regime[(ma60.isna()) | (ret20.isna())] = "unknown"
    normalized_dates = pd.to_datetime(dates).dt.normalize()
    return pd.Series(regime.reindex(normalized_dates).to_numpy(dtype=object), index=dates.index)


def _cross_section_regime_source(
    daily_long: pd.DataFrame,
    *,
    date_col: str,
    sym_col: str,
) -> pd.DataFrame:
    df = daily_long.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df[sym_col] = df[sym_col].astype(str).str.zfill(6)
    close_wide = df.pivot(index=date_col, columns=sym_col, values="close").sort_index().astype(np.float64)
    return pd.DataFrame(
        {
            date_col: close_wide.index,
            "close": close_wide.mean(axis=1, skipna=True).to_numpy(dtype=np.float64),
        }
    )


def _assign_score_quantiles(panel: pd.DataFrame, *, n_quantiles: int) -> pd.DataFrame:
    out = panel.copy()
    q = max(int(n_quantiles), 2)
    q = min(q, len(out))
    ranked = out["score"].rank(method="first")
    labels = [f"Q{i}" for i in range(1, q + 1)]
    out["bucket"] = pd.qcut(ranked, q=q, labels=labels)
    return out


def _score_quantile_edges(scores: pd.Series, *, n_quantiles: int) -> list[float]:
    clean = scores.astype(np.float64).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return []
    q = min(max(int(n_quantiles), 2), len(clean))
    raw_edges = np.nanquantile(clean.to_numpy(dtype=np.float64), np.linspace(0.0, 1.0, q + 1))
    edges = np.unique(raw_edges.astype(np.float64))
    if len(edges) < 2:
        return []
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.tolist()


def _stack_wide(df: pd.DataFrame) -> pd.Series:
    try:
        return df.stack(future_stack=True)
    except TypeError:
        return df.stack(dropna=False)
