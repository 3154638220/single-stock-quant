"""Capital allocation across ranked BUY candidates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def allocate_top_n(
    scores: pd.DataFrame,
    *,
    n_top: int = 5,
    max_per_stock: float = 0.25,
    min_score: float = 0.0,
    min_volume_rank: int | None = None,
    volume_wide: pd.DataFrame | None = None,
    daily_long: pd.DataFrame | None = None,
    date_col: str = "trade_date",
    sym_col: str = "symbol",
) -> pd.DataFrame:
    """Pick top N stocks by daily score and assign equal weights.

    Returns a wide DataFrame (index=trade_date, columns=symbol) with weights
    summing to at most 1.0 per day.
    """
    w = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)

    # Liquidity filter: exclude stocks with < 100M avg daily amount
    if min_volume_rank is not None and volume_wide is not None:
        vol_allowed = _liquidity_mask(
            daily_long, volume_wide,
            date_col=date_col, sym_col=sym_col, min_amount=min_volume_rank,
        )
    else:
        vol_allowed = pd.DataFrame(True, index=scores.index, columns=scores.columns)

    for dt in scores.index:
        day_scores = scores.loc[dt]
        allowed = day_scores.index[day_scores >= min_score]
        if vol_allowed is not None and dt in vol_allowed.index:
            vmask = vol_allowed.loc[dt] if dt in vol_allowed.index else pd.Series(True, index=allowed)
            allowed = allowed[vmask.reindex(allowed).fillna(False).astype(bool)]
        if len(allowed) == 0:
            continue
        top = day_scores[allowed].nlargest(n_top)
        top = top[top > 0]
        if top.empty:
            continue
        raw = np.clip(1.0 / len(top), 0.0, float(max_per_stock))
        # Normalise so sum <= 1.0
        total = raw * len(top)
        if total > 1.0:
            raw /= total
        w.loc[dt, top.index] = raw

    return w


def allocate_volatility_weighted(
    scores: pd.DataFrame,
    *,
    n_top: int = 5,
    max_per_stock: float = 0.25,
    min_score: float = 0.0,
    atr_lookback: int = 20,
    close_wide: pd.DataFrame | None = None,
    high_wide: pd.DataFrame | None = None,
    low_wide: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Allocate top N stocks with inverse-volatility weights.

    Lower recent ATR → larger weight (up to *max_per_stock*).
    """
    w = allocate_top_n(
        scores, n_top=n_top, max_per_stock=max_per_stock, min_score=min_score,
    )
    if w.empty or close_wide is None:
        return w

    for dt in w.index:
        active = w.columns[w.loc[dt] > 0]
        if len(active) <= 1:
            continue
        inv_vol = {}
        for sym in active:
            if sym not in close_wide.columns:
                inv_vol[sym] = 1.0
                continue
            c = close_wide[sym].loc[:dt].dropna().tail(atr_lookback + 1)
            if high_wide is not None and sym in high_wide.columns:
                h = high_wide[sym].loc[:dt].dropna().tail(atr_lookback + 1)
                l = low_wide[sym].loc[:dt].dropna().tail(atr_lookback + 1)
            else:
                h, l = c * 1.01, c * 0.99
            prev_c = c.shift(1)
            tr = pd.concat(
                [h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1
            ).max(axis=1)
            atr = tr.rolling(14, min_periods=14).mean().iloc[-1]
            vol = atr / c.iloc[-1] if c.iloc[-1] > 0 else 1.0
            inv_vol[sym] = 1.0 / max(vol, 0.001)
        total_inv = sum(inv_vol.values())
        if total_inv <= 0:
            continue
        for sym in active:
            w.loc[dt, sym] = min(inv_vol.get(sym, 1.0) / total_inv, float(max_per_stock))
        # Normalise
        row_sum = float(w.loc[dt].sum())
        if row_sum > 1.0:
            w.loc[dt] /= row_sum

    return w


def _liquidity_mask(
    daily_long: pd.DataFrame | None,
    volume_wide: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    sym_col: str = "symbol",
    min_amount: int = 100_000_000,
) -> pd.DataFrame:
    """Filter out symbols with average daily amount below *min_amount*."""
    if daily_long is None:
        return pd.DataFrame(True, index=volume_wide.index, columns=volume_wide.columns)
    df = daily_long.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df[sym_col] = df[sym_col].astype(str).str.zfill(6)
    px = pd.to_numeric(df.get("close", df.get("open", pd.Series(0, index=df.index))), errors="coerce")
    vol = pd.to_numeric(df.get("volume", pd.Series(0, index=df.index)), errors="coerce")
    amount = px * vol
    df["_amount"] = amount
    avg_amount = df.pivot(index=date_col, columns=sym_col, values="_amount").sort_index().rolling(20, min_periods=20).mean()
    return avg_amount >= float(min_amount)


def apply_constraints(
    weights: pd.DataFrame,
    *,
    max_positions: int = 5,
    max_per_stock: float = 0.25,
    max_per_industry: float = 0.40,
    industry_map: dict[str, str] | None = None,
    max_daily_turnover: float = 0.50,
) -> pd.DataFrame:
    """Apply portfolio constraints to a weights DataFrame.

    - Enforce max positions per day
    - Cap individual stock weights
    - Cap industry exposure
    - Limit daily turnover
    """
    w = weights.copy().astype(np.float64)
    w = w.clip(upper=float(max_per_stock))

    # Max positions
    for dt in w.index:
        row = w.loc[dt]
        active = row[row > 0]
        if len(active) > max_positions:
            keep = active.nlargest(max_positions).index
            w.loc[dt, ~w.columns.isin(keep)] = 0.0

    # Industry constraint
    if industry_map is not None:
        for dt in w.index:
            industry_weights: dict[str, float] = {}
            for sym in w.columns:
                if w.loc[dt, sym] > 0:
                    ind = industry_map.get(sym, sym)
                    industry_weights[ind] = industry_weights.get(ind, 0.0) + w.loc[dt, sym]
            for ind, iw in industry_weights.items():
                if iw > max_per_industry:
                    scale = max_per_industry / iw
                    for sym in w.columns:
                        if industry_map.get(sym, sym) == ind:
                            w.loc[dt, sym] *= scale

    # Turnover constraint
    if max_daily_turnover < 1.0:
        prev = None
        for dt in w.index:
            row = w.loc[dt]
            if prev is not None:
                turnover = 0.5 * float(np.sum(np.abs(row.to_numpy(dtype=np.float64) - prev.to_numpy(dtype=np.float64))))
                if turnover > max_daily_turnover:
                    scale = max_daily_turnover / turnover
                    w.loc[dt] = prev + (row - prev) * scale
            prev = row.copy()

    return w
