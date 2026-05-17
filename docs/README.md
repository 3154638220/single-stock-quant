# Documentation

Current documentation is scoped to single-stock quantitative trading. Historical
archive entries may mention old indicator-replication work; those entries are
not current project goals.

- [indicator_formula.md](indicator_formula.md): DK trend formulas and parameters.
- [backtest_guide.md](backtest_guide.md): single-stock backtest assumptions and CLI examples.

Completed plans are archived in [archive/](archive/) for reference.

For a local visual check, run:

```bash
pip install ".[viz]"
python scripts/plot_dktrend.py --symbol 600930 --history 180
```
