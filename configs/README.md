# Configs

The active configuration template is the repository root `config.yaml.example`.

The project goal is single-stock trading. Configs with legacy names such as
`eastmoney_dkbar` are research configs for DKBar-style trading signals; they are
not Eastmoney indicator replication targets.

The key section is `trend_signal`:

```yaml
trend_signal:
  mode: macd_cross
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  ma_fast: 5
  ma_slow: 20
  ma_smooth: 3
  boll_window: 20
```
