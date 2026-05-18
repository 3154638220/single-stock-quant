"""Trend indicators for single-stock timing strategies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .utils import ema, ema_seeded

try:  # Optional acceleration; the pure-Python path below remains the fallback.
    from numba import njit
except Exception:  # pragma: no cover - depends on optional runtime package
    njit = None


class TrendMode(str, Enum):
    MACD_CROSS = "macd_cross"
    MA_CROSS = "ma_cross"
    BOLL_TREND = "boll_trend"
    DONCHIAN_BREAKOUT = "donchian_breakout"
    EASTMONEY_DKBAR = "eastmoney_dkbar"
    LONG_MA_TREND = "long_ma_trend"
    DUAL_MA_CROSS = "dual_ma_cross"
    TREND_SCORE = "trend_score"


@dataclass(frozen=True)
class DKTrendParams:
    mode: TrendMode | str = TrendMode.MACD_CROSS
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ma_fast: int = 5
    ma_slow: int = 20
    ma_smooth: int = 3
    boll_window: int = 20
    min_run_len: int = 1
    donchian_entry_window: int = 20
    donchian_exit_window: int = 10
    # DKBar-style trading signal params. The legacy mode name is kept for
    # compatibility; this project does not target official indicator replication.
    lst_period: int = 180
    lst_method: str = "ema"
    bar_period: int = 10
    bar_method: str = "ema"
    bar_color_method: str = "price_change"
    bar_color_hold_days: int = 3
    bar_color_min_red_run: int = 2
    bar_range_period: int = 5
    bar_range_mult: float = 0.10
    slope_tolerance: float = 0.0
    state_confirm_days: int = 2
    hysteresis_pct: float = 0.003
    atr_period: int = 14
    atr_mult: float = 2.5
    dkx_signal_period: int = 10
    # Long MA trend params
    trend_ma_period: int = 250
    trend_ma_type: str = "ema"
    slope_lookback: int = 10
    require_positive_slope: bool = True
    # Dual MA cross params
    dual_ma_fast: int = 30
    dual_ma_slow: int = 120
    # Trend score params
    trend_score_ma_long: int = 250
    trend_score_ma_fast: int = 30
    trend_score_ma_slow: int = 120
    # Signal confirmation
    min_breakout_days: int = 3

    @classmethod
    def from_mapping(cls, data: dict | None) -> "DKTrendParams":
        d = dict(data or {})
        raw_mode = d.get("mode", TrendMode.MACD_CROSS.value)
        return cls(
            mode=raw_mode if isinstance(raw_mode, TrendMode) else TrendMode(str(raw_mode)),
            macd_fast=int(d.get("macd_fast", 12)),
            macd_slow=int(d.get("macd_slow", 26)),
            macd_signal=int(d.get("macd_signal", 9)),
            ma_fast=int(d.get("ma_fast", 5)),
            ma_slow=int(d.get("ma_slow", 20)),
            ma_smooth=int(d.get("ma_smooth", 3)),
            boll_window=int(d.get("boll_window", 20)),
            min_run_len=int(d.get("min_run_len", 1)),
            donchian_entry_window=int(d.get("donchian_entry_window", 20)),
            donchian_exit_window=int(d.get("donchian_exit_window", 10)),
            lst_period=int(d.get("lst_period", 180)),
            lst_method=str(d.get("lst_method", "ema")),
            bar_period=int(d.get("bar_period", 10)),
            bar_method=str(d.get("bar_method", "ema")),
            bar_color_method=str(d.get("bar_color_method", "price_change")),
            bar_color_hold_days=int(d.get("bar_color_hold_days", 3)),
            bar_color_min_red_run=int(d.get("bar_color_min_red_run", 2)),
            bar_range_period=int(d.get("bar_range_period", 5)),
            bar_range_mult=float(d.get("bar_range_mult", 0.10)),
            slope_tolerance=float(d.get("slope_tolerance", 0.0)),
            state_confirm_days=int(d.get("state_confirm_days", 2)),
            hysteresis_pct=float(d.get("hysteresis_pct", 0.003)),
            atr_period=int(d.get("atr_period", 14)),
            atr_mult=float(d.get("atr_mult", 2.5)),
            dkx_signal_period=int(d.get("dkx_signal_period", 10)),
            trend_ma_period=int(d.get("trend_ma_period", 250)),
            trend_ma_type=str(d.get("trend_ma_type", "ema")),
            slope_lookback=int(d.get("slope_lookback", 10)),
            require_positive_slope=bool(d.get("require_positive_slope", True)),
            dual_ma_fast=int(d.get("dual_ma_fast", 30)),
            dual_ma_slow=int(d.get("dual_ma_slow", 120)),
            trend_score_ma_long=int(d.get("trend_score_ma_long", 250)),
            trend_score_ma_fast=int(d.get("trend_score_ma_fast", 30)),
            trend_score_ma_slow=int(d.get("trend_score_ma_slow", 120)),
            min_breakout_days=int(d.get("min_breakout_days", 3)),
        )


def _trend_mode(value: TrendMode | str) -> TrendMode:
    if isinstance(value, TrendMode):
        return value
    return TrendMode(str(value))


def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("ohlcv is empty")
    missing = {"close"} - set(df.columns)
    if missing:
        raise ValueError(f"ohlcv missing required columns: {sorted(missing)}")
    out = df.copy()
    if "trade_date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.normalize()
        out = out.sort_values("trade_date").set_index("trade_date", drop=False)
    else:
        out.index = pd.to_datetime(out.index).normalize()
        out = out.sort_index()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out


def _indicator_value(df: pd.DataFrame, params: DKTrendParams) -> pd.Series:
    close = pd.to_numeric(df["close"], errors="coerce")
    mode = _trend_mode(params.mode)
    if mode == TrendMode.MACD_CROSS:
        fast = ema(close, params.macd_fast)
        slow = ema(close, params.macd_slow)
        diff = fast - slow
        dea = ema(diff, params.macd_signal)
        return diff - dea
    if mode == TrendMode.MA_CROSS:
        ma_fast = close.rolling(params.ma_fast, min_periods=params.ma_fast).mean()
        ma_slow = close.rolling(params.ma_slow, min_periods=params.ma_slow).mean()
        return ema(ma_fast - ma_slow, params.ma_smooth)
    if mode == TrendMode.BOLL_TREND:
        middle = close.rolling(params.boll_window, min_periods=params.boll_window).mean()
        return close - middle
    if mode == TrendMode.EASTMONEY_DKBAR:
        return _compute_eastmoney_dkbar(df, params)["dk_value"]
    if mode == TrendMode.LONG_MA_TREND:
        trend_line = _compute_ma(close, params.trend_ma_period, params.trend_ma_type)
        slope = trend_line.diff(params.slope_lookback)
        raw = close - trend_line
        if params.require_positive_slope:
            raw = raw.where(slope > 0, -raw.abs())
        return raw
    if mode == TrendMode.DUAL_MA_CROSS:
        fast = _compute_ma(close, params.dual_ma_fast, params.trend_ma_type)
        slow = _compute_ma(close, params.dual_ma_slow, params.trend_ma_type)
        return fast - slow
    if mode == TrendMode.TREND_SCORE:
        return _trend_score(df, close, params)
    raise ValueError(f"unsupported trend mode: {params.mode}")


def _compute_ma(series: pd.Series, period: int, ma_type: str) -> pd.Series:
    if ma_type == "ema":
        return ema_seeded(series, period)
    if ma_type == "wma":
        return series.rolling(period, min_periods=period).apply(
            lambda x: ((x * (np.arange(len(x)) + 1)) / (np.arange(len(x)) + 1).sum()).sum()
        )
    if ma_type == "hull":
        wma_half = _wma(series, period // 2)
        wma_full = _wma(series, period)
        hma_raw = 2 * wma_half - wma_full
        return _wma(hma_raw, int(np.sqrt(period)))
    return ema(series, period)


def _smooth(series: pd.Series, period: int, method: str) -> pd.Series:
    method_l = str(method).lower()
    p = max(int(period), 1)
    if method_l in {"ema", "seeded_ema"}:
        return ema_seeded(series, p)
    if method_l == "sma":
        return pd.to_numeric(series, errors="coerce").rolling(p, min_periods=p).mean()
    if method_l == "wma":
        return _wma(series, p)
    if method_l == "kama":
        return _kama(series, p)
    return ema_seeded(series, p)


def _kama(series: pd.Series, period: int, fast: int = 2, slow: int = 30) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    p = max(int(period), 1)
    change = (s - s.shift(p)).abs()
    volatility = s.diff().abs().rolling(p, min_periods=p).sum()
    er = (change / volatility).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    values = s.to_numpy(dtype=float)
    sc_values = sc.to_numpy(dtype=float)
    result = np.full(len(s), np.nan, dtype=float)
    seed = s.rolling(p, min_periods=p).mean()
    first_valid = seed.first_valid_index()
    if first_valid is None:
        return pd.Series(result, index=s.index, dtype=float)
    first_pos = s.index.get_loc(first_valid)
    result[first_pos] = float(seed.iloc[first_pos])
    for i in range(first_pos + 1, len(values)):
        if np.isfinite(values[i]) and np.isfinite(result[i - 1]):
            result[i] = result[i - 1] + sc_values[i] * (values[i] - result[i - 1])
    return pd.Series(result, index=s.index, dtype=float)


def _typical_price(df: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df.get("open", close), errors="coerce")
    high = pd.to_numeric(df.get("high", close), errors="coerce")
    low = pd.to_numeric(df.get("low", close), errors="coerce")
    return (3.0 * close + low + open_ + high) / 6.0


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df.get("high", close), errors="coerce")
    low = pd.to_numeric(df.get("low", close), errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(max(int(period), 1), min_periods=max(int(period), 1)).mean()


def _dkx_line(df: pd.DataFrame) -> pd.Series:
    typical = _typical_price(df)
    weights = np.arange(20, 0, -1, dtype=float)
    return typical.rolling(20, min_periods=20).apply(lambda x: float(np.dot(x, weights) / weights.sum()), raw=True)


def _supertrend_lst(df: pd.DataFrame, params: DKTrendParams) -> pd.Series:
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df.get("high", close), errors="coerce")
    low = pd.to_numeric(df.get("low", close), errors="coerce")
    atr = _atr(df, params.atr_period)
    hl2 = (high + low) / 2.0
    upper = hl2 + float(params.atr_mult) * atr
    lower = hl2 - float(params.atr_mult) * atr
    result = np.full(len(df), np.nan, dtype=float)
    trend = ""
    for i in range(len(df)):
        c = close.iloc[i]
        if not np.isfinite(c) or not np.isfinite(upper.iloc[i]) or not np.isfinite(lower.iloc[i]):
            continue
        if trend == "":
            trend = "red" if c >= hl2.iloc[i] else "green"
        elif trend == "red" and c < result[i - 1]:
            trend = "green"
        elif trend == "green" and c > result[i - 1]:
            trend = "red"
        if trend == "red":
            prev = result[i - 1] if i > 0 and np.isfinite(result[i - 1]) else lower.iloc[i]
            result[i] = max(float(lower.iloc[i]), float(prev))
        else:
            prev = result[i - 1] if i > 0 and np.isfinite(result[i - 1]) else upper.iloc[i]
            result[i] = min(float(upper.iloc[i]), float(prev))
    return pd.Series(result, index=df.index, dtype=float)


def _bar_mid(df: pd.DataFrame, params: DKTrendParams) -> pd.Series:
    method = str(params.bar_method).lower()
    if method == "dkx":
        return _dkx_line(df)
    if method == "heikin":
        close = pd.to_numeric(df["close"], errors="coerce")
        open_ = pd.to_numeric(df.get("open", close), errors="coerce")
        high = pd.to_numeric(df.get("high", close), errors="coerce")
        low = pd.to_numeric(df.get("low", close), errors="coerce")
        heikin_close = (open_ + high + low + close) / 4.0
        return _smooth(heikin_close, params.bar_period, "ema")
    return _smooth(_typical_price(df), params.bar_period, method)


def _lst_line(df: pd.DataFrame, params: DKTrendParams) -> pd.Series:
    method = str(params.lst_method).lower()
    if method == "supertrend":
        return _supertrend_lst(df, params)
    if method == "dkx_signal":
        return _smooth(_dkx_line(df), params.dkx_signal_period, "sma")
    return _smooth(_typical_price(df), params.lst_period, method)


def _confirm_colors(
    raw_red: pd.Series,
    raw_green: pd.Series,
    dk_value: pd.Series,
    *,
    confirm_days: int,
) -> pd.Series:
    confirm = max(int(confirm_days), 1)
    colors: list[str] = []
    state = ""
    red_count = 0
    green_count = 0
    for idx in raw_red.index:
        is_red = bool(raw_red.loc[idx]) if pd.notna(raw_red.loc[idx]) else False
        is_green = bool(raw_green.loc[idx]) if pd.notna(raw_green.loc[idx]) else False
        if is_red and not is_green:
            red_count += 1
            green_count = 0
        elif is_green and not is_red:
            green_count += 1
            red_count = 0
        else:
            red_count = 0
            green_count = 0

        if state == "":
            v = dk_value.loc[idx]
            if red_count >= confirm:
                state = "red"
            elif green_count >= confirm:
                state = "green"
            elif pd.notna(v):
                state = "red" if float(v) > 0 else "green"
        elif state == "red" and green_count >= confirm:
            state = "green"
        elif state == "green" and red_count >= confirm:
            state = "red"
        colors.append(state)
    return pd.Series(colors, index=raw_red.index, dtype="object")


def _transition_signals(colors: pd.Series, min_run_len: int = 1) -> tuple[pd.Series, pd.Series]:
    run_len = _run_lengths(colors)
    prev_color = colors.shift(1)
    min_run = max(int(min_run_len), 1)
    signal = pd.Series("", index=colors.index, dtype="object")
    if min_run <= 1:
        signal.loc[(colors == "red") & (prev_color == "green")] = "buy"
        signal.loc[(colors == "green") & (prev_color == "red")] = "sell"
    else:
        prev_run_len = run_len.shift(1).fillna(0).astype("int64")
        signal.loc[(colors == "red") & (run_len >= min_run) & (prev_run_len < min_run)] = "buy"
        signal.loc[(colors == "green") & (run_len >= min_run) & (prev_run_len < min_run)] = "sell"
    return signal, run_len


def _visual_bar_color(
    df: pd.DataFrame,
    bar_mid: pd.Series,
    lst: pd.Series,
    trend_state: pd.Series,
    params: DKTrendParams,
) -> pd.Series:
    method = str(params.bar_color_method).lower()
    close = pd.to_numeric(df["close"], errors="coerce")
    if method == "bar_slope":
        delta = bar_mid.diff()
        color = pd.Series("", index=df.index, dtype="object")
        color.loc[delta > 0] = "red"
        color.loc[delta < 0] = "green"
        color.loc[delta == 0] = trend_state.loc[delta == 0]
        return color
    if method == "trend_state":
        return trend_state.copy()
    if method == "persistent_price_change":
        return _persistent_price_change_color(df, bar_mid, lst, trend_state, params)

    prev_close = close.shift(1)
    color = pd.Series("", index=df.index, dtype="object")
    color.loc[close >= prev_close] = "red"
    color.loc[close < prev_close] = "green"
    color.loc[prev_close.isna() & close.notna()] = trend_state.loc[prev_close.isna() & close.notna()]
    return color


def _persistent_price_change_color(
    df: pd.DataFrame,
    bar_mid: pd.Series,
    lst: pd.Series,
    trend_state: pd.Series,
    params: DKTrendParams,
) -> pd.Series:
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    hold_days = max(int(params.bar_color_hold_days), 0)
    min_red_run = max(int(params.bar_color_min_red_run), 1)
    hyst = float(params.hysteresis_pct)

    if _persistent_color_numba is not None:
        trend_code = trend_state.map({"red": 1, "green": -1}).fillna(0).to_numpy(dtype=np.int64)
        codes = _persistent_color_numba(
            close.to_numpy(dtype=np.float64),
            prev_close.to_numpy(dtype=np.float64),
            pd.to_numeric(bar_mid, errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(lst, errors="coerce").to_numpy(dtype=np.float64),
            trend_code,
            hold_days,
            min_red_run,
            hyst,
        )
        return pd.Series(
            np.where(codes == 1, "red", np.where(codes == -1, "green", "")),
            index=df.index,
            dtype="object",
        )

    color = pd.Series("", index=df.index, dtype="object")
    red_run = 0
    pullback_days = 0
    prev_color = ""
    for idx in df.index:
        c = close.loc[idx]
        pc = prev_close.loc[idx]
        if pd.isna(c):
            color.loc[idx] = ""
            red_run = 0
            pullback_days = 0
            prev_color = ""
            continue
        if pd.isna(pc):
            current = trend_state.loc[idx] if trend_state.loc[idx] in ("red", "green") else ""
            pullback_days = 0
        elif c >= pc:
            current = "red"
            pullback_days = 0
        else:
            strong_red_trend = (
                prev_color == "red"
                and red_run >= min_red_run
                and trend_state.loc[idx] == "red"
                and pd.notna(bar_mid.loc[idx])
                and pd.notna(lst.loc[idx])
                and float(bar_mid.loc[idx]) > float(lst.loc[idx]) * (1.0 + hyst)
            )
            pullback_days = pullback_days + 1 if prev_color == "red" else 1
            current = "red" if strong_red_trend and pullback_days <= hold_days else "green"

        if current == "red":
            red_run = red_run + 1 if prev_color == "red" else 1
        else:
            red_run = 0
        if current != "red":
            pullback_days = 0
        color.loc[idx] = current
        prev_color = current
    return color


if njit is not None:
    @njit(cache=True)
    def _persistent_color_numba(
        close_arr,
        prev_close_arr,
        bar_mid_arr,
        lst_arr,
        trend_state_arr,
        hold_days,
        min_red_run,
        hyst,
    ):
        out = np.zeros(len(close_arr), dtype=np.int64)
        red_run = 0
        pullback_days = 0
        prev_color = 0
        for i in range(len(close_arr)):
            c = close_arr[i]
            pc = prev_close_arr[i]
            if not np.isfinite(c):
                out[i] = 0
                red_run = 0
                pullback_days = 0
                prev_color = 0
                continue
            if not np.isfinite(pc):
                current = trend_state_arr[i] if trend_state_arr[i] == 1 or trend_state_arr[i] == -1 else 0
                pullback_days = 0
            elif c >= pc:
                current = 1
                pullback_days = 0
            else:
                strong_red_trend = (
                    prev_color == 1
                    and red_run >= min_red_run
                    and trend_state_arr[i] == 1
                    and np.isfinite(bar_mid_arr[i])
                    and np.isfinite(lst_arr[i])
                    and bar_mid_arr[i] > lst_arr[i] * (1.0 + hyst)
                )
                pullback_days = pullback_days + 1 if prev_color == 1 else 1
                current = 1 if strong_red_trend and pullback_days <= hold_days else -1

            if current == 1:
                red_run = red_run + 1 if prev_color == 1 else 1
            else:
                red_run = 0
            if current != 1:
                pullback_days = 0
            out[i] = current
            prev_color = current
        return out
else:
    _persistent_color_numba = None


def _compute_eastmoney_dkbar(df: pd.DataFrame, params: DKTrendParams) -> pd.DataFrame:
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df.get("high", close), errors="coerce")
    low = pd.to_numeric(df.get("low", close), errors="coerce")

    lst = _lst_line(df, params)
    mid = _bar_mid(df, params)
    raw_range = (high - low).abs()
    width = _smooth(raw_range, params.bar_range_period, "ema") * float(params.bar_range_mult)
    bar_high = mid + width / 2.0
    bar_low = mid - width / 2.0
    slope = lst.diff(max(int(params.slope_lookback), 1))
    tol = float(params.slope_tolerance)
    hyst = float(params.hysteresis_pct)
    upper_trigger = lst * (1.0 + hyst)
    lower_trigger = lst * (1.0 - hyst)
    raw_red = (bar_low > upper_trigger) & (slope >= -tol)
    raw_green = (bar_high < lower_trigger) | (slope < -tol)
    value = mid - lst
    trend_state = _confirm_colors(raw_red.fillna(False), raw_green.fillna(False), value, confirm_days=params.state_confirm_days)
    trend_state.loc[value.isna()] = ""
    signal, trend_run_len = _transition_signals(trend_state, params.min_run_len)
    bar_color = _visual_bar_color(df, mid, lst, trend_state, params)
    bar_color.loc[value.isna()] = ""
    bar_run_len = _run_lengths(bar_color)

    return pd.DataFrame(
        {
            "dk_value": value.astype(float),
            "dk_color": trend_state,
            "dk_signal": signal,
            "dk_run_len": trend_run_len,
            "bar_color": bar_color,
            "bar_run_len": bar_run_len,
            "trend_state": trend_state,
            "lst": lst.astype(float),
            "bar_high": bar_high.astype(float),
            "bar_low": bar_low.astype(float),
            "bar_mid": mid.astype(float),
            "trend_run_len": trend_run_len,
        },
        index=df.index,
    )


def _wma(series: pd.Series, period: int) -> pd.Series:
    p = max(1, int(period))
    return series.rolling(p, min_periods=p).apply(
        lambda x: ((x * (np.arange(len(x)) + 1)) / (np.arange(len(x)) + 1).sum()).sum(),
        raw=True,
    )


def _trend_score(df: pd.DataFrame, close: pd.Series, params: DKTrendParams) -> pd.Series:
    ma_long = _compute_ma(close, params.trend_score_ma_long, params.trend_ma_type)
    ma_fast = _compute_ma(close, params.trend_score_ma_fast, params.trend_ma_type)
    ma_slow = _compute_ma(close, params.trend_score_ma_slow, params.trend_ma_type)
    cond_price_above_long = (close > ma_long).astype(float)
    cond_fast_above_slow = (ma_fast > ma_slow).astype(float)
    score = 0.5 * cond_price_above_long + 0.5 * cond_fast_above_slow
    return (score - 0.5) * 2


def _run_lengths(colors: pd.Series) -> pd.Series:
    run = []
    prev = None
    n = 0
    for color in colors:
        if color not in ("red", "green"):
            run.append(0)
            prev = None
            n = 0
            continue
        if color == prev:
            n += 1
        else:
            n = 1
            prev = color
        run.append(n)
    return pd.Series(run, index=colors.index, dtype="int64")


def compute_dktrend(df: pd.DataFrame, params: DKTrendParams | None = None) -> pd.DataFrame:
    """
    Add ``dk_value``, ``dk_color``, ``dk_signal`` and ``dk_run_len`` columns.

    ``dk_color`` is red when the selected trend value is positive, otherwise green.
    Signals are emitted only on red/green transitions after the initial warmup span.
    """
    p = params or DKTrendParams()
    out = _validate_ohlcv(df)
    if _trend_mode(p.mode) == TrendMode.EASTMONEY_DKBAR:
        trend = _compute_eastmoney_dkbar(out, p)
        for col in trend.columns:
            out[col] = trend[col]
        return out.replace({np.nan: np.nan})

    value = _indicator_value(out, p)
    valid = value.notna()
    color = pd.Series("", index=out.index, dtype="object")
    color.loc[valid & (value > 0)] = "red"
    color.loc[valid & (value <= 0)] = "green"

    signal, run_len = _transition_signals(color, p.min_run_len)

    out["dk_value"] = value.astype(float)
    out["dk_color"] = color
    out["dk_signal"] = signal
    out["dk_run_len"] = run_len
    return out.replace({np.nan: np.nan})
