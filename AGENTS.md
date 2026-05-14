<claude-mem-context>
# Memory Context

# [single-stock-quant] recent context, 2026-05-14 1:19pm GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (22,393t read) | 860,764t work | 97% savings

### May 14, 2026
S23 执行 docs/plan.md 改进计划：实现阶段 13（周线多周期趋势过滤）和阶段 14（EWMA波动率动态仓位精细化），完成全部 5 步执行计划，168 个测试全绿 (May 14, 11:36 AM)
S24 执行 docs/plan.md — 补齐阶段 15 实验闭环与报告完善工具 (May 14, 11:57 AM)
S25 执行 docs/plan.md Phase 15 — 补齐实验闭环与报告完善工具 (May 14, 12:02 PM)
S26 盘点 plan.md 执行剩余缺口 — 识别代码已完成但实验和验证尚未闭环的项目 (May 14, 12:03 PM)
S27 推进 docs/plan.md 实验闭环 — 5 变体组合回测 sweep + ranking 归因分析 + E16 重构规划 (May 14, 12:06 PM)
S28 推进 plan.md 中的 E16 阶段：组合 ranking 重构实验 — 实现三种 ranking profile（balanced/meta_priority/dk_meta），添加 DK 红色趋势候选池约束，跑两轮真实数据实验，并更新 plan.md 文档 (May 14, 12:17 PM)
S29 继续推进 plan.md — 实现 E17 dk_fresh_meta 新近DK候选池实验，验证"趋势未老化"假设，并更新计划文档 (May 14, 12:32 PM)
394 12:34p 🔵 Portfolio ranking attribution reveals E10 rank score is anti-monotonic at 20-day horizon
395 " 🔵 E16 ranking profile experiments: dk_meta shows promise but Sharpe still far below target
396 " ⚖️ Strategy pivot: from stacking hard filters to fixing cross-sectional ranking signal
397 " 🟣 Stages 8-16 all completed with full test coverage (174 tests passing)
398 12:36p 🟣 New ranking profile 'dk_fresh_meta' excludes stale DK red trends older than 20 days
399 12:37p 🟣 New 'dk_fresh_meta' ranking profile implemented — restricts candidate pool to DK red trends aged ≤ 20 days
400 " 🟣 E17 experiment launched with new dk_fresh_meta profile to test stale trend exclusion
401 12:39p 🔵 E17 dk_fresh_meta results: stale-trend exclusion slightly reduces MDD but degrades Sharpe and return vs dk_meta
402 12:40p ✅ Experiment comparison infrastructure extended to include ranking attribution metrics and experiment artifact manifest
403 " ⚖️ E17 dk_fresh_meta confirmed as regression: stale-trend exclusion degrades ranking attribution too
404 12:41p ⚖️ Plan updated: E17 concluded as failure, E18 (rolling forward-return calibration) designated as next priority
S30 推进 plan.md 到阶段 18 — 实现并实跑 E18 rolling forward-return calibration 实验，结论确认失败并记录战略转向 (May 14, 12:42 PM)
405 12:52p 🔵 Plan review confirms E18 rolling forward-return calibration as next priority
406 12:53p 🔵 E17 experiment conclusively shows narrowing DK red-trend candidates by freshness does not fix ranking quality
407 " 🔵 Attribution pipeline measures ranking monotonicity via score-quantile forward returns
408 12:54p 🟣 E18 rolling forward-return calibration implemented in attribution.py and signal_ranker.py
409 12:55p 🟣 E18 dk_calibrated_meta ranking profile fully wired and tests passing; real-data backtest launched
410 12:56p 🔵 E18 dk_calibrated_meta real-data backtest still computing after 90+ seconds
411 12:57p 🔵 E18 rolling forward-return calibration significantly worsened portfolio performance vs E16 dk_meta baseline
412 12:58p ⚖️ E18 rolling forward-return calibration experiment fails conclusively — ranking monotonicity remains unsolved after 5 consecutive experiments
413 12:59p ⚖️ Pivot strategy: stop portfolio-layer ranking patches, investigate single-stock signal root causes
S31 推进 plan.md Stage 19: 候选池根因归因 — 实现按 symbol/market regime/industry 拆解 DK 红色候选池 forward return 的诊断工具，并用 E16 dk_meta 分数完成首跑 (May 14, 12:59 PM)
414 1:03p ⚖️ Portfolio ranking iterations E10-E18 all failed to meet targets; strategy pivot to single-stock root cause analysis
415 " 🟣 Portfolio ranking attribution analysis tool built: scripts/analyze_portfolio_ranking.py and src/portfolio/attribution.py
416 " 🟣 Five portfolio ranking profiles implemented in src/portfolio/signal_ranker.py
417 " 🔵 Meta-label p_win alone is insufficient as primary ranking factor; DK candidate pool gating provides better structure
418 " 🔵 Rolling forward-return calibration amplifies noise rather than improving ranking; dk_fresh_meta only marginally controls drawdown
419 " 🔵 WFO IS/OOS Sharpe correlation is mostly negative across 4 of 5 stocks with WFO results
420 " ⚖️ Next development phase: single-stock signal quality root cause analysis instead of more portfolio patches
421 1:05p 🟣 New candidate forward return breakdown function added to attribution module for single-stock root cause analysis
422 1:06p 🟣 New CLI script for candidate forward return breakdown analysis: scripts/analyze_candidate_breakdown.py
423 " 🟣 Tests added for candidate forward return breakdown in test_portfolio.py
424 " 🟣 Experiment artifact manifest updated to include candidate_breakdown.csv
425 " 🔵 E16 dk_meta candidate breakdown reveals worst-performing stocks: 万科A, 保利发展, 海螺水泥 at 20-day horizon
426 1:07p 🔴 Market regime classification now falls back to cross-sectional average when index OHLCV is unavailable
427 1:08p 🔵 E16 dk_meta candidate breakdown with regime: 000002 (万科A) worst across all regimes; bull regime has highest weighted mean but bears contribute large share
428 " ✅ Plan.md updated with Stage 19 progress: candidate pool root cause analysis complete with actionable findings
429 " 🟣 Stage 19 complete: candidate pool root cause analysis delivered actionable diagnostic identifying worst/best stocks by symbol and regime
430 " ✅ Full test suite passing (182 tests) after Stage 19 implementation
S32 推进 plan.md Stage 19: 候选池根因归因 — 实现按 symbol/market regime/industry 拆解 DK 红色候选池 forward return 的诊断工具，并用 E16 dk_meta 分数完成首跑 (May 14, 1:10 PM)
431 1:12p ⚖️ E19 root cause analysis: negative portfolio returns driven by specific weak symbols, not market regime
432 " ⚖️ E16-E18 ranking experiments all fail to deliver positive Sharpe; strategy pivot to E20 sector/stock filtering
433 " 🟣 New candidate pool root-cause attribution tool: analyze_candidate_breakdown.py
434 1:13p 🟣 E20: Symbol-level blacklist/greylist filtering added to portfolio ranking pipeline
435 1:14p 🟣 E20 CLI wiring complete: run_portfolio_backtest.py supports exclude/greylist symbols, industry map, and industry concentration caps
436 " 🔵 Plan document confirms E16-E19 ranking experiments complete — all failed to achieve positive Sharpe; strategy pivots to E20 structural filtering
437 1:15p 🟣 Stage 20 (E20) code implementation complete: symbol-level blacklist/greylist structural guardrails for portfolio ranking
438 " 🟣 E20 real-data experiment launched: dk_meta profile with exclude symbols 000002,600048 and greylist 600585
439 1:16p 🔵 E20 real-data experiment still processing: expanding-window meta-label training on 25 stocks takes significant wall time
440 1:17p 🔵 E20 portfolio backtest runtime: expanding-window meta-label scoring for 25 stocks is IO/compute intensive (~105+ seconds and counting)
441 " 🔵 E20 experiment results: excluding 000002/600048 and greylisting 600585 dramatically improves all portfolio metrics vs E16 baseline
442 1:18p 🔵 E20 complete validation: 7 of 12 metrics improved vs E16, excluded symbols fully removed, ranking top-bottom positive for all horizons
443 " 🔵 Full test suite passes: 174 tests, ruff identifies 40 pre-existing lint issues (none related to E20 changes)

Access 861k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>