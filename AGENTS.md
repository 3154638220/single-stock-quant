<claude-mem-context>
# Memory Context

# [single-stock-quant] recent context, 2026-05-16 9:52pm GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (11,342t read) | 208,869t work | 95% savings

### May 15, 2026
S44 User provided 6 Eastmoney screenshot anchor points for stock 000783 — calibration revealed bar_color ≠ trade_state, leading to architectural pivot and full implementation of bar_color/trend_state split across the codebase (May 15, 9:22 PM)
S45 User provided 6 Eastmoney screenshot anchor points for stock 000783 — calibration disproved the bar_color=trade_state hypothesis, leading to an architectural split of visual bar_color from trading trend_state, with full implementation across code/configs/tests/docs (May 15, 9:23 PM)
S46 Claude clarified to user: the Eastmoney DK bar indicator has NOT been successfully replicated — only the engineering framework was corrected (bar_color/trend_state split), and the actual indicator reverse-engineering is still unsolved (May 15, 9:23 PM)
S47 User asked if there's a next-phase plan — Claude confirmed the plan document has direction (bar_color/trend_state split done, need to optimize each independently) but lacks a detailed executable task list (May 15, 9:24 PM)
S48 User provided 6 Eastmoney screenshot anchor points for stock 000783 (dates, LST, BARHIGH, BARLOW, color) — Claude calibrated the DK bar indicator against them, discovered the bar_color ≠ trade_state hypothesis was fundamentally wrong, made an architectural pivot, and fully implemented the bar_color/trend_state split across code, configs, tests, and documentation (May 15, 9:24 PM)
S49 Advance plan-05-15(2).md: Implement E21 (anchor calibration dataset solidification) and E22 (bar_color state machine reverse-engineering) for Eastmoney DK Trend replication on stock 000783 (May 15, 9:30 PM)
S50 Advance plan-05-15(2).md: Implement E21 (anchor calibration dataset solidification) and E22 (bar_color state machine reverse-engineering) for Eastmoney DK Trend replication on stock 000783, then run grid search to discover optimal parameters (May 15, 9:43 PM)
S51 Advance plan-05-15(2).md through E21 and E22 implementation, then pivot strategy from anchor-first to return-first calibration for Eastmoney DK Trend reverse-engineering on stock 000783 (May 15, 9:53 PM)
S52 诊断000783长江证券DK趋势背测问题：5笔交易总收益+20.61%但依赖单笔大赢家，分析入场信号、退出机制、参数稳定性 (May 15, 9:54 PM)
### May 16, 2026
S53 诊断000783长江证券DK趋势背测：5笔交易总收益+20.61%但依赖单笔大赢家，分析入场信号滞后、退出保护缺失、参数稳定性不足三大问题 (May 16, 8:46 PM)
595 9:12p 🔴 Fix in progress: _stability np.float64 crash on string params replaced with pd.to_numeric
596 9:13p 🔴 _stability string crash fixed and composite score constraints made configurable
597 " 🔴 WFO config loosened for low-frequency DK Bar strategy: trades/year 0.5-8, max drawdown 40%
598 " 🟣 Test added for _stability categorical parameter handling: test_stability_accepts_categorical_params
599 9:14p 🔵 All 17 tests pass after _stability fix; WFO run for 000783 DK Bar in progress
600 " 🔵 WFO 000783 DK Bar run still computing — 576 param combos × multiple folds expected to be long-running
601 " 🔵 WFO 000783 DK Bar: ~55 seconds elapsed, still computing with no output — 576 combos processing
602 " 🔵 WFO 000783 DK Bar completed but produced zero trades — all metrics NaN/zero across 2 folds
603 9:15p 🔵 DK Bar WFO 000783: In-sample trades exist but OOS generates zero trades across all 576 combos
604 9:17p 🔵 DK Bar trend mode: compute_dktrend delegates to _compute_eastmoney_dkbar when mode is EASTMONEY_DKBAR
605 " 🟣 OOS backtest now uses train-warmed trend override to prevent cold-start zero-trade issue
606 " 🟣 Test added for _oos_trend_with_warmup: test_oos_trend_uses_train_rows_as_warmup
607 9:18p 🔵 All 11 tests in test_wfo_params_split.py pass including new warmup test; WFO 000783 DK Bar re-run launched
608 " 🔵 WFO 000783 re-run with warmup fix: ~40 seconds elapsed, still computing 576 combos × 2 folds
609 " 🔵 WFO 000783 re-run: ~65 seconds into computation, still processing — expected for 1152 backtests
610 9:19p 🔵 WFO 000783 re-run: ~75 seconds elapsed, prior run completed at this time — results imminent
611 " 🔴 WFO 000783 DK Bar warmup fix confirmed: OOS now produces 29.57% return, Sharpe 0.94, MDD 10.84%
612 " 🔵 Full-sample 000783 DK Bar backtest: 5 trades, +30.17% total, Sharpe 0.52, avg hold 45 days
613 9:20p ✅ WFO 000783 DK Bar re-run with --export-results launched; single backtest confirmed 5 trades across full period
614 " 🔵 WFO 000783 with --export-results: ~50 seconds elapsed, still computing — expected for 576 combos × 2 folds
615 " 🔵 WFO 000783 export run: ~65 seconds elapsed, still processing — results imminent based on prior ~85-90s runtime
616 9:21p 🔵 WFO 000783 export run: ~75 seconds elapsed, approaching prior completion time of ~88 seconds
617 " 🔵 WFO 000783 export run: ~85 seconds elapsed — results expected any moment now
618 " ✅ WFO 000783 DK Bar exported: results JSON written to data/output/000783_wfo_20260516.json
619 9:22p ✅ Full DK Bar grid search launched: calibrate_eastmoney_dkbar.py for 000783 with --grid-search --sort-by return_quality
620 " 🔵 Grid search calibrate_eastmoney_dkbar.py for 000783: ~30 seconds elapsed, still computing full parameter space
621 " 🔵 Grid search confirms WFO-selected params: sma205_wma5_c3_h0 ranks #1 by return_quality with 30.17% return
622 9:23p 🔵 Full test suite passes 100% — all changes backward-compatible across 5 modified files
623 9:41p 🟣 WFO 字符串参数稳定性统计修复完成
624 " 🟣 退出层峰值保护和 ATR trailing 止损语义补齐并加测试
625 " ✅ 000783 对照回测与 DK WFO 验证重跑完成
626 9:42p ✅ 计划文档同步更新：intrapos_dd_limit=0.06 显著提升回测表现
627 " ⚖️ intrapos_dd_limit=0.06 成为回测配置的标准风控参数
628 " 🔵 WFO OOS warmup 修复使长均线策略的 OOS 评估变得可靠
629 9:43p ✅ 计划文档完成最终同步：E24/E25 里程碑更新，下一步进入 5 股小样本验证
630 " 🔵 Trade 3 具体修复：intrapos_dd_stop 将 -1.13% 扭转为 +6.71%
631 9:44p 🔵 跨股 WFO 验证启动：600030 运行中，600036 因 DuckDB 锁冲突失败
632 " ⚖️ 5股小样本跨股WFO验证启动，600030率先运行，600036需等待DuckDB锁释放
633 9:45p 🔵 600030（中信证券）DK WFO 失败：OOS Sharpe -1.15，参数漂移 0.913
634 " ⚖️ 600030 WFO 失败不阻断跨股验证，600036（招商银行）WFO 继续串行执行
635 " 🔵 600036（招商银行）DK WFO 同样失败：OOS Sharpe -0.05，参数漂移 0.816
636 9:46p 🔵 跨股验证第三只标的 000568（泸州老窖）WFO 启动
637 9:47p 🔵 000568（泸州老窖）DK WFO 也失败：OOS Sharpe -0.83，三只跨股标的连续未通过
638 " ⚖️ 跨股验证扩展至第四只标的 300750（宁德时代），测试创业板新能源股
639 9:48p 🔵 300750（宁德时代）WFO 出现两次运行结果：第二次 OOS Sharpe 达 0.47 且参数漂移为零
640 " ✅ 第五只跨股验证标的 600519（贵州茅台）WFO 启动
641 9:50p 🔵 600519（贵州茅台）DK WFO 也失败：OOS Sharpe -1.53，5只跨股标的全部未通过
642 9:51p 🔵 跨股WFO汇总分析：6只股票仅000783通过，300750两个fold参数完全相同
643 " ⚖️ 跨股WFO验证结论：DK策略不具备跨股泛化能力，不能进入25股扩展
644 " ✅ 计划文档全面更新完成：跨股WFO结果、小样本汇总表、E25方向调整全部写入

Access 209k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>