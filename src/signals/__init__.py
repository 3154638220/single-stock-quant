from .consensus import compute_consensus_trend, generate_consensus_signals
from .generator import SignalRecord, generate_signals, get_current_signal
from .types import Position, Signal

__all__ = [
    "Position",
    "Signal",
    "SignalRecord",
    "compute_consensus_trend",
    "generate_consensus_signals",
    "generate_signals",
    "get_current_signal",
]
