import numpy as np

from src.market.tradability import (
    is_open_limit_up_unbuyable,
    is_row_suspended_like,
    limit_up_px,
    limit_up_ratio,
)


def test_limit_up_ratio_by_board():
    assert limit_up_ratio("600000") == 0.10
    assert limit_up_ratio("000001") == 0.10
    assert limit_up_ratio("300750") == 0.20
    assert limit_up_ratio("688001") == 0.20
    assert limit_up_ratio("830000") == 0.30


def test_open_limit_up_unbuyable_uses_board_ratio_and_tolerance():
    assert limit_up_px(10.0, "600000") == 11.0
    assert is_open_limit_up_unbuyable(10.999, 10.0, "600000")
    assert not is_open_limit_up_unbuyable(10.95, 10.0, "600000")
    assert is_open_limit_up_unbuyable(12.0, 10.0, "300750")
    assert not is_open_limit_up_unbuyable(11.5, 10.0, "300750")


def test_open_limit_up_treats_invalid_prices_as_unbuyable():
    assert is_open_limit_up_unbuyable(np.nan, 10.0, "600000")
    assert is_open_limit_up_unbuyable(11.0, np.nan, "600000")
    assert is_open_limit_up_unbuyable(11.0, 0.0, "600000")


def test_suspended_like_rows():
    assert not is_row_suspended_like(1000, 10.0, 10.2)
    assert is_row_suspended_like(0, 10.0, 10.2)
    assert is_row_suspended_like(np.nan, 10.0, 10.2)
    assert is_row_suspended_like(1000, np.nan, 10.2)
