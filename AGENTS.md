<claude-mem-context>
# Memory Context

# [single-stock-quant] recent context, 2026-05-16 9:09pm GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (19,168t read) | 832,733t work | 98% savings

### May 15, 2026
S44 User provided 6 Eastmoney screenshot anchor points for stock 000783 — calibration revealed bar_color ≠ trade_state, leading to architectural pivot and full implementation of bar_color/trend_state split across the codebase (May 15, 9:22 PM)
S45 User provided 6 Eastmoney screenshot anchor points for stock 000783 — calibration disproved the bar_color=trade_state hypothesis, leading to an architectural split of visual bar_color from trading trend_state, with full implementation across code/configs/tests/docs (May 15, 9:23 PM)
S46 Claude clarified to user: the Eastmoney DK bar indicator has NOT been successfully replicated — only the engineering framework was corrected (bar_color/trend_state split), and the actual indicator reverse-engineering is still unsolved (May 15, 9:23 PM)
S47 User asked if there's a next-phase plan — Claude confirmed the plan document has direction (bar_color/trend_state split done, need to optimize each independently) but lacks a detailed executable task list (May 15, 9:24 PM)
S48 User provided 6 Eastmoney screenshot anchor points for stock 000783 (dates, LST, BARHIGH, BARLOW, color) — Claude calibrated the DK bar indicator against them, discovered the bar_color ≠ trade_state hypothesis was fundamentally wrong, made an architectural pivot, and fully implemented the bar_color/trend_state split across code, configs, tests, and documentation (May 15, 9:24 PM)
539 9:28p ✅ Plan document completely rewritten with structured executive summary and concrete E21–E25 execution roadmap
540 " ✅ Plan document completely rewritten: ~960-line accumulated doc → clean 7-section ~400-line structured roadmap
S49 Advance plan-05-15(2).md: Implement E21 (anchor calibration dataset solidification) and E22 (bar_color state machine reverse-engineering) for Eastmoney DK Trend replication on stock 000783 (May 15, 9:30 PM)
541 9:32p 🔵 DK Trend reverse-engineering: engineering structure separates visual layer from trading layer
542 " ⚖️ 5-phase execution plan (E21-E25) defined with clear gating criteria before expanding to 25 stocks
543 " 🔵 7 anchor points for 000783 with known Eastmoney ground-truth values for LST, BARHIGH, BARLOW, and bar_color
544 " 🔵 Working tree shows 10 modified files and 2 new config files across indicators, backtest, scripts, and tests
545 9:34p 🟣 E21 started: created data/anchors/ directory to hold calibration anchor CSV
546 " 🟣 E21 complete: anchor CSV, calibration scoring script, and validation tests implemented
547 " 🔵 Calibration script reveals config defaults produce empty bar_color for all 7 anchors; sma60_wma20 is best candidate at 5/7 hit rate
548 9:35p 🔴 Calibration script patched: bar_color_accuracy now counts all anchors and default start extended to 2020-01-01
549 9:36p 🔵 Diagnostic probe of failing anchors reveals bar_color flips green on intra-trend price dips while bar_mid remains above LST
550 " 🔵 Multi-window probe confirms bar_color=price_change flips green on every down day regardless of trend context
551 9:37p 🟣 E22 implemented: persistent_price_change state machine added to bar_color system in dktrend.py
552 9:38p 🟣 Test added for persistent_price_change: validates red-hold on pullbacks vs price_change baseline
553 " 🔵 persistent_price_change produces zero improvement over price_change on real data — all 8 candidates still at 5/7
554 9:39p 🔴 persistent_price_change bug fixed: pullback_days counter now resets on price rise and initial state
555 " 🔵 Bug-fixed persistent_price_change achieves 6/7 anchor hit rate with persistent_sma60_wma20 — E22 acceptance criteria met
556 9:40p 🔴 Persistent-price-change unit test parameters corrected to match working grid result
557 " ✅ Unit test parameters corrected: lst_period=5, bar_period=1 with wider closes grid for persistent_price_change test
558 " 🟣 E22 complete: all 7 tests pass, persistent_sma60_wma20 achieves 6/7 anchor hit rate meeting ≥6/7 acceptance criteria
559 9:41p ✅ Plan document updated to reflect E21 completion and E22 partial success with persistent_sma60_wma20 at 6/7
560 " ✅ Plan document fully updated: E21 and E22 sections marked complete with results, E22 gating clarified for E23
561 9:42p 🟣 Full test suite passes (all tests green) after E21+E22 implementation — 7 key files modified, 3 new files created
562 9:43p ✅ Gitignore updated to track anchor CSVs while keeping data directory excluded
S50 Advance plan-05-15(2).md: Implement E21 (anchor calibration dataset solidification) and E22 (bar_color state machine reverse-engineering) for Eastmoney DK Trend replication on stock 000783, then run grid search to discover optimal parameters (May 15, 9:43 PM)
563 9:46p 🟣 Calibration script upgraded with grid search, backtest integration, and multi-objective ranking
564 9:49p 🔵 Grid search across 120 DK bar candidates: best objective_score is -0.36, all negative — trend_state drives trading not bar_color
565 9:50p ✅ Grid search optimized: persistent_price_change removed from grid as it produces identical backtest results to price_change
566 " 🔵 Full grid search over 252 candidates finds sma205_wma5_c3_h0 achieves +0.372 objective score with +23.86% return and 4 trades — first positive score in the session
567 9:51p ✅ Both DK bar configs updated to sma205_wma5 with persistent_price_change, confirm_days=3, hysteresis=0.0 — WFO param grid now uses DK-specific dimensions
568 9:52p ⚖️ Plan strategy shifted: anchor fit demoted from optimization objective to visual constraint; return quality now primary ranking criterion
S51 Advance plan-05-15(2).md through E21 and E22 implementation, then pivot strategy from anchor-first to return-first calibration for Eastmoney DK Trend reverse-engineering on stock 000783 (May 15, 9:53 PM)
S52 诊断000783长江证券DK趋势背测问题：5笔交易总收益+20.61%但依赖单笔大赢家，分析入场信号、退出机制、参数稳定性 (May 15, 9:54 PM)
### May 16, 2026
569 8:34p 🔵 DK Trend背测分析：000783长江证券5笔交易深度诊断
570 8:35p 🔵 DK趋势退出机制优先级链确认：profit_lock在ATR trailing之后执行
571 " 🔵 eastmoney_dkbar配置显式关闭profit_lock，与生产配置形成对比
572 " 🔵 DK参数网格搜索确认SMA205/WMA5/state_confirm=3为000783最优收益候选
573 " 🔵 背测退出在000783上time_stop主导，ATR trailing仅在Trade 4成功触发
574 " 🔵 背测参数通过config.py统一构建，确保single/batch/WFO入口一致性
575 " 🔵 eastmoney_dkbar校准流程：锚点评分+背测收益双维度评估候选参数
576 8:37p 🔵 场景测试揭示：profit_lock产生反效果，降低000783总收益从+20.6%到+8.3%
577 " 🔵 000783网格搜索：仅23/252候选正收益，SMA205是唯一正收益LST周期
578 " 🔵 000783 WFO stable已通过MACD模式获OOS Sharpe 0.81，DK模式WFO正在运行中
579 " 🔵 退出机制场景全面对比：intrapos_dd_limit=0.06为最优，dk_fade_exit显著有害
580 " 🔵 Trade 3浮盈转亏的微观机制：价格从7.84峰值回落至6.68，但ATR在高峰时仅0.28
581 8:39p 🔵 WFO crash：_stability函数无法处理DK参数网格中的字符串类型参数
582 8:40p 🔵 校准脚本完成：confirm=3是收益最优配置的充分条件，confirm=1/2/3的收益呈单调递增
583 " 🔵 入场质量过滤器对000783效果分析：min_quality_score=50剔除了Trade 2但降低了总收益
584 8:41p 🔵 入场过滤器的所有动量过滤器都过滤掉了唯一盈利交易Trade 4
585 8:42p 🔵 Pullback入场过滤器系统性损害000783收益：总收益从+20.6%降至+0~2%
586 " ⚖️ 000783 DK趋势策略诊断结论：问题不在参数选择，而在标的本身趋势持续性差且样本量过小
587 8:43p 🔵 Profit_lock网格搜索确认：高触发阈值(0.25/0.04)为唯一改善组合，低阈值全部损害收益
588 " 🔵 Profit_lock在000783上的阈值分界线精确确定：trigger≥0.18且trailing≥0.06时从不触发
S53 诊断000783长江证券DK趋势背测：5笔交易总收益+20.61%但依赖单笔大赢家，分析入场信号滞后、退出保护缺失、参数稳定性不足三大问题 (May 16, 8:46 PM)
**Investigated**: 1. **逐笔交易微观追踪**：检查了5笔交易的入场DK指标（LST/bar_mid/dk_value/trend_state）、持仓期内dk_signal变化、价格峰谷与ATR关系
    2. **退出机制全面场景测试**：profit_lock 35组参数网格（trigger 0.08~0.25 × trailing 0.04~0.12）、ATR trailing 1.0~3.0、intrapos_dd 5%~15%、dk_fade 2~5天、time_stop开/关、多种组合
    3. **入场过滤器测试**：min_quality_score 50/55、require_rs60、require_ma120、weekly_bullish、volume_ratio_min 1.5/2.0、pullback_entry 1~8天
    4. **DK参数网格搜索**：252候选聚合分析（按LST周期/方法分组、confirm_days对比、hysteresis_pct效果）
    5. **DK模式WFO尝试**：发现_stability函数对字符串参数崩溃；自定义绕过脚本在OOS窗口返回NaN
    6. **校准脚本完整运行**：return_quality排序top15确认sma205_wma5_c3_h0最优
    7. **2026年4-5月逐日DK数据**：确认DK indicators正常计算（之前NaN是start='2026-04-01'数据窗口不足SMA205）
    8. **代码级机制分析**：退出优先级链（src/backtest/single_stock.py行720-770）、入场质量评分7维度（src/signals/generator.py行80-162）、DK bar计算逻辑（src/indicators/dktrend.py行438-470）

**Learned**: 1. **入场信号滞后是核心问题**：DK buy信号在2024-10-10触发时价格已从924行情+100%涨幅后开始回落（当天跌-8.04%），Trade 5也是在连续两个+10%涨停后才触发。SMA205+confirm=3的3天确认期放大了滞后
    2. **profit_lock对000783是双刃剑**：trigger=0.08/trailing=0.04使Trade 3改善(+8.2%)但Trade 4暴跌(+2.6%)，净负。trigger=0.18为分界线——低于此值触发Trade 3的profit_lock；trigger=0.25/trailing=0.04为唯一改善组合(+26.56%, Sharpe 0.47)
    3. **所有入场动量过滤器都排除Trade 4**：Trade 4入场时RS60=-4.86%，RS60>0/MA120之上/周线看涨/quality>55全部剔除这笔唯一盈利交易。反转入场与动量过滤器天然矛盾
    4. **参数鲁棒性极差**：252候选中仅23个(9.1%)正收益，中位数-20.85%。仅SMA205一个LST周期产生正平均收益，bar_period=5优于10，confirm=3优于1/2
    5. **Trade 4入场的独特性**：2025-04-24 trend_state由green翻红产生buy信号，入场时price≈LST(dist_lst仅0.02%)，是趋势反转早期入场点
    6. **ATR trailing语义限制**：行730要求"当前收益仍≥8%"才检查ATR trailing，导致Trade 3在浮盈从+16%回落后ATR trailing被跳过，time_stop在46天时才接管
    7. **intrapos_dd=6%为本样本最佳退出**：总收益+30.17%、Sharpe 0.52，Trade 3在+6.7%被保护退出。但高度可能过拟合
    8. **MACD WFO在000783上OOS Sharpe 0.81**：说明问题在DK模式适配性而非标的不可交易

**Completed**: 1. 完成5笔交易逐日DK指标追踪，确认Trade 2/3持仓期内无任何dk_signal
    2. 完成19种退出机制场景测试，确认intrapos_dd=6%最优、dk_fade完全有害
    3. 完成35组profit_lock网格搜索，精确确定阈值分界线
    4. 完成8组入场过滤器测试，确认识别出"所有过滤器排除Trade 4"的悖论
    5. 完成5组pullback_entry场景测试，确认延迟入场全部损害收益
    6. 完成252候选DK参数网格聚合分析
    7. 完成return_quality校准脚本运行
    8. 发现并记录DK WFO两处工程问题（_stability崩溃、OOS NaN）
    9. 输出最终诊断结论给用户

**Next Steps**: 按用户最终结论的优先级：
    1. **修退出层**：实现"峰值收益达阈值后按持仓峰值回撤退出"规则，加成熟度约束避免Trade 4早期小回撤被误触发。intrapos_dd=6%需WFO验证
    2. **修入场层**：新增过热过滤——close/LST距离、近N日涨幅、ATR%上限、连续大阳后禁入。当前momentum过滤器无效因排除反转交易
    3. **修复DK WFO工程**：_stability函数支持字符串参数；解决OOS窗口无有效交易问题
    4. **优先推荐**：对000783使用已验证的MACD WFO模式（OOS Sharpe 0.81），DK模式仅作为辅助参考


Access 833k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>