"""Tests for sector constraint logic in rotation backtest."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.rotation import _apply_sector_constraint, run_rotation_backtest
from src.indicators import DKTrendParams, TrendMode


class TestApplySectorConstraint:
    def test_one_per_sector_with_two_same_sector(self):
        ranked = [("300750", 1.5), ("002594", 1.2), ("600036", 0.9)]
        sector_map = {"300750": "新能源", "002594": "新能源", "600036": "银行"}
        result = _apply_sector_constraint(ranked, sector_map, top_n=2)
        assert result == {"300750", "600036"}  # 300750 gets the 新能源 slot

    def test_all_different_sectors(self):
        ranked = [("300750", 1.5), ("600036", 1.2), ("600519", 0.9)]
        sector_map = {"300750": "新能源", "600036": "银行", "600519": "白酒"}
        result = _apply_sector_constraint(ranked, sector_map, top_n=3)
        assert result == {"300750", "600036", "600519"}

    def test_unmapped_symbol_gets_own_sector(self):
        ranked = [("300750", 1.5), ("999999", 1.2)]
        sector_map = {"300750": "新能源"}
        result = _apply_sector_constraint(ranked, sector_map, top_n=2)
        assert result == {"300750", "999999"}  # 999999 treated as its own sector

    def test_top_n_limits_selection(self):
        ranked = [("300750", 1.5), ("600036", 1.2), ("600519", 0.9)]
        sector_map = {"300750": "新能源", "600036": "银行", "600519": "白酒"}
        result = _apply_sector_constraint(ranked, sector_map, top_n=1)
        assert result == {"300750"}  # only top-1

    def test_negative_scores_excluded_by_caller(self):
        # _apply_sector_constraint doesn't filter by score — caller does
        ranked = [("300750", -1.0), ("600036", -2.0)]
        sector_map = {"300750": "新能源", "600036": "银行"}
        result = _apply_sector_constraint(ranked, sector_map, top_n=2)
        assert result == {"300750", "600036"}  # passes through regardless of score
