<claude-mem-context>
# Memory Context

# [single-stock-quant] recent context, 2026-05-15 8:05pm GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (21,039t read) | 619,090t work | 97% savings

### May 14, 2026
S27 推进 docs/plan.md 实验闭环 — 5 变体组合回测 sweep + ranking 归因分析 + E16 重构规划 (May 14, 12:06 PM)
S28 推进 plan.md 中的 E16 阶段：组合 ranking 重构实验 — 实现三种 ranking profile（balanced/meta_priority/dk_meta），添加 DK 红色趋势候选池约束，跑两轮真实数据实验，并更新 plan.md 文档 (May 14, 12:17 PM)
S29 继续推进 plan.md — 实现 E17 dk_fresh_meta 新近DK候选池实验，验证"趋势未老化"假设，并更新计划文档 (May 14, 12:32 PM)
S30 推进 plan.md 到阶段 18 — 实现并实跑 E18 rolling forward-return calibration 实验，结论确认失败并记录战略转向 (May 14, 12:42 PM)
S31 推进 plan.md Stage 19: 候选池根因归因 — 实现按 symbol/market regime/industry 拆解 DK 红色候选池 forward return 的诊断工具，并用 E16 dk_meta 分数完成首跑 (May 14, 12:59 PM)
414 1:03p ⚖️ Portfolio ranking iterations E10-E18 all failed to meet targets; strategy pivot to single-stock root cause analysis
415 " 🟣 Portfolio ranking attribution analysis tool built: scripts/analyze_portfolio_ranking.py and src/portfolio/attribution.py
416 " 🟣 Five portfolio ranking profiles implemented in src/portfolio/signal_ranker.py
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
S32 推进 plan.md Stage 19: 候选池根因归因 — 实现按 symbol/market regime/industry 拆解 DK 红色候选池 forward return 的诊断工具，并用 E16 dk_meta 分数完成首跑 (May 14, 1:09 PM)
S33 推进 E20：结构性 symbol/行业护栏 — 实现、实跑、验证并回写 plan.md，基于 E19 根因分析对 000002/600048 排除、600585 降权 (May 14, 1:10 PM)
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
444 1:19p 🟣 E20 complete: Stage 20 structural filtering implemented and validated — first experiment (E16-E20) to achieve all positive portfolio metrics
S34 Analyzed plan-05-15(2).md improvement plan and comparison charts to assess whether the new long-period MA trend indicators can replicate East Money's "多空趋势" orange curve (May 14, 1:20 PM)
### May 15, 2026
445 7:32p 🔵 DK indicator is fundamentally wrong: MACD histogram ≠ trend line
446 " 🔵 EMA cold-start bias corrupts first 3-6 months of long-period trend signals
447 " 🔵 Exit signal lags price peak by 16-26 days due to triple EMA chain
448 " 🔵 profit_lock creates systematic friction on low-priced stocks, cutting winners at bottoms
449 " 🔵 Strategy success entirely dependent on one unrepeatable policy event (2024-09 stimulus)
450 " 🔵 WFO optimizes wrong parameter space: macd_fast/macd_signal noise instead of trend period
451 " ⚖️ Three hypotheses for East Money's actual trend algorithm, with price-vs-EMA as most likely
452 " ⚖️ 6-phase improvement roadmap: redesign indicator → visual verification → exit mechanism → WFO → backtest → scale
453 " ⚖️ Visual verification success criteria: orange curve position, switch count ≤ 5/year, LST anchor match
S35 Corrected the replication target: East Money's "多空趋势" is a main-chart overlay with LST orange trend line + BARHIGH/BARLOW red-green bars, not a sub-chart MACD histogram (May 15, 7:34 PM)
454 7:36p 🔵 Project's intellectual history: from assuming DK=MACD histogram to recognizing fundamental indicator error
455 7:51p 🔵 Root cause analysis: DK trend indicator fundamentally mismatches target
456 7:52p ⚖️ Three new trend modes proposed for dktrend.py: LONG_MA_TREND, DUAL_MA_CROSS, TREND_SCORE
457 " ⚖️ Exit mechanism redesigned: trend-line-based exit replaces profit_lock + MACD-dependent sell
458 " ⚖️ WFO parameter space redesigned: trend_ma_period replaces macd_fast/macd_signal as optimization target
459 " ⚖️ Visual comparison tool specified: plot_dktrend.py enhanced with multi-mode overlay and switch-frequency stats
460 7:54p ⚖️ Plan corrected: target is red-green bar state machine, not just orange trend line
461 " ⚖️ Reverse-engineering hypotheses completely rewritten with 5 new hypotheses and concrete calibration anchor
462 " ⚖️ EASTMONEY_DKBAR mode specified: structured DataFrame output with LST, bar fields, state machine, and hysteresis
463 7:55p ⚖️ DKTrendParams restructured: 10 new EASTMONEY_DKBAR-specific fields with hysteresis and state confirmation
464 " ⚖️ Visual tool redesigned: plot_eastmoney_dkbar() replaces plot_comparison() with red/green vlines and LST overlay
465 7:56p ⚖️ Exit mechanism and WFO both realigned to eastmoney_dkbar color-flip signals
466 " ✅ Plan document fully aligned: all stale "long_ma_trend/trend line" references replaced with eastmoney_dkbar terminology
467 7:57p ✅ Plan document fully edited: all sections now correctly target eastmoney_dkbar red-green bar state machine
S36 Revise improvement plan docs/plan-05-15(2).md to correct target from "replicate orange trend line" to "replicate 东方财富 multi-empty red-green bar state machine with LST/BARHIGH/BARLOW" (May 15, 7:58 PM)
**Investigated**: The 799-line plan document was read in 5 chunks (lines 1-220, 220-520, 520-700, 720-805, 180-460, 460-740) to understand the full scope before editing. A 2026-02-03 screenshot calibration anchor was discovered with exact values: close=8.46, daily range 8.36-8.76, but 东方财富 LST=7.66, BARHIGH=8.38, BARLOW=8.35 — proving BARHIGH/BARLOW are not OHLC copies but indicator-internal compressed bars. Five reverse-engineering hypotheses were formulated (A through E), with Hypothesis A (LST long-period baseline + bar state machine) ranked highest priority.

**Learned**: The core insight is that 东方财富's "多空趋势" is a structured main-chart overlay with three components: (1) an orange LST baseline line acting as a trend floor or trailing stop, (2) red/green bars bounded by BARHIGH/BARLOW that are compressed smooth-price bars (not daily OHLC), and (3) a state machine with hysteresis that prevents frequent color flips. The current code outputs a single MACD histogram scalar — wrong indicator type AND wrong output structure. Trading signals should come from bar_color state transitions (green→red = buy, red→green = sell), not from MACD histogram sign changes.

**Completed**: Nine patches were applied to docs/plan-05-15(2).md, comprehensively revising every section:
    - Title and core goal corrected to target red-green bar replication
    - New Section Zero added with 8-field structured output spec (lst, bar_high, bar_low, bar_mid, bar_color, signal, run_len)
    - Reverse-engineering section rewritten with 5 hypotheses and 2026-02-03 calibration anchor
    - Phase A.1 changed from LONG_MA_TREND to EASTMONEY_DKBAR as primary mode; LONG_MA_TREND/DUAL_MA_CROSS demoted to baseline
    - DKTrendParams restructured with 10 new EASTMONEY_DKBAR-specific fields including hysteresis (state_confirm_days=2, hysteresis_pct=0.003)
    - Visual tool redesigned: plot_eastmoney_dkbar() draws orange LST line + red/green vlines from BARLOW to BARHIGH
    - Exit mechanism changed from "close below trend line" to "dk_color_changed_to_green()"
    - WFO config renamed wfo_eastmoney_dkbar.yaml with 135-combo grid (5×3×3×3)
    - Phase E acceptance adds numerical anchor check and ≥80% color consistency
    - Sections 10-12 (priority table, stop-doing list, success criteria, next steps) all aligned to eastmoney_dkbar terminology
    - Final grep confirmed 0 stale references; code fence count verified even at 52
    - File status: new untracked file (?? docs/plan-05-15(2).md)

**Next Steps**: Phase A.1 is the immediate next step: implement TrendMode.EASTMONEY_DKBAR in src/indicators/dktrend.py, outputting the 8-column DataFrame (dk_value, dk_color, dk_signal, lst, bar_high, bar_low, bar_mid, trend_run_len). Then Phase B.1: run the visual tool to calibrate against the 2026-02-03 screenshot anchor. Finally, connect bar_color state transitions to the backtest signal pipeline.


Access 619k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>