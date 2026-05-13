"""Signal and position enums."""

from __future__ import annotations

from enum import Enum


class Signal(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class Position(str, Enum):
    LONG = "long"
    FLAT = "flat"


class SignalQuality(str, Enum):
    WEAK = "weak"
    FAIR = "fair"
    STRONG = "strong"
