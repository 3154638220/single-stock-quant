---
name: experiment-results-round6
description: Key findings from Round 6 (plan-05-19) experiments executed 2026-05-19
metadata:
  type: project
---

## E-2: Rotation Meta-Parameter Cross-Period Validation

**Finding**: `top_n=1, freq=10, trend_strength` is the most robust (avg_ann_rank=5.0/27, best in 4/4 periods).
**Why**: Current default `top_n=2, freq=10` was NOT in top 1/3 for any ranking mode. `top_n=1` best in all 4 sub-periods.
**How to apply**: Use `top_n=1, rebalance_freq=10, trend_strength` as default parameters. Avoid `top_n=2` unless specific evidence supports it.

## E-3: Ranking Factor IC Analysis

**Finding**: Only `dk_value` has meaningful positive ICIR (0.123). Momentum (rs_20, rs_60) and above_ma120 all have negative ICIR.
**Why**: A-share market does not exhibit momentum effects in this stock pool. Multi-factor ranking that includes momentum dilutes the signal.
**How to apply**: `multi_factor_score()` updated: w_trend=0.85, w_vol_adj=0.15. Removed momentum, trend_dir, run_len components. Keep `trend_strength` as default ranking mode.

## E-2 + E-3 Combined

`multi_factor` consistently underperforms `trend_strength` across all periods. Per plan decision node: "放弃 multi_factor，保持 trend_strength".
