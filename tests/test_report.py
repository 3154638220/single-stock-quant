import pandas as pd

from src.backtest.report import generate_html_report
from src.backtest.single_stock import run_single_stock_backtest
from src.indicators import DKTrendParams, TrendMode


def _flat_df(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100] * len(closes),
        }
    )


class TestHTMLReport:
    def test_generate_html_report_returns_string(self):
        closes = [10, 10, 10, 10, 11, 12, 13, 14]
        df = _flat_df(closes)
        res = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
            cost_bps=0, initial_capital=10000,
        )
        html = generate_html_report(res, df)
        assert "<html" in html.lower()
        assert "600000" in html

    def test_generate_html_report_writes_to_disk(self, tmp_path):
        closes = [10, 10, 10, 10, 11, 12, 13, 14]
        df = _flat_df(closes)
        res = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
            cost_bps=0, initial_capital=10000,
        )
        out = tmp_path / "report.html"
        html = generate_html_report(res, df, output_path=str(out))
        assert out.exists()
        assert "600000" in out.read_text(encoding="utf-8")

    def test_generate_html_report_with_index_ohlcv(self):
        closes = [10, 10, 10, 10, 11, 12, 13, 14]
        df = _flat_df(closes)
        index_df = pd.DataFrame(
            {
                "symbol": ["510300"] * len(closes),
                "trade_date": pd.date_range("2024-01-01", periods=len(closes)),
                "open": [100] * len(closes),
                "high": [100] * len(closes),
                "low": [100] * len(closes),
                "close": [100, 101, 102, 103, 104, 105, 106, 107],
                "volume": [100] * len(closes),
            }
        )
        res = run_single_stock_backtest(
            "600000", df,
            DKTrendParams(mode=TrendMode.BOLL_TREND, boll_window=3),
            cost_bps=0, initial_capital=10000,
        )
        html = generate_html_report(res, df, index_ohlcv=index_df)
        assert "CSI300" in html or "牛市" in html or "熊市" in html or "震荡" in html
