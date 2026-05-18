<claude-mem-context>
# Memory Context

# [single-stock-quant] recent context, 2026-05-18 4:36pm GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (16,313t read) | 1,646,275t work | 99% savings

### May 17, 2026
654 9:41p 🔵 2026-05-16 最新 WFO 结果确认：仅 2/6 只标的 Sharpe 为正，泛化能力严重不足
655 " 🔵 生产环境配置 prod-v1 仅使用 4 只 WFO 通过的标的，Meta-label 因 600030 上劣化被禁用
656 " 🔵 calibrate_eastmoney_dkbar.py 的 objective_score 公式权重硬编码且缺乏校准依据
657 " 🔵 persistent_price_change 颜色状态机在纯 Python 循环中逐行计算，性能瓶颈明显
661 9:43p 🔵 代码质量扫描：ruff 发现 43 个问题（30 个可自动修复），mypy 发现 13 个类型错误
662 " 🔵 WFO JSON 输出存在 NaN 序列化问题——heatmap 数据包含大量 NaN 值
663 " 🔵 000783 DK WFO 的参数选择揭示了 platform 机制的实际效果：两个 Fold 均未选峰值
664 9:48p 🔵 新任务启动：系统化更新项目目标表述和文档措辞
665 " 🔵 全项目"复刻/东财"关键词搜索揭示目标表述散布在 30+ 个文件中
666 " 🔵 docs/indicator_formula.md 和 configs/README.md 同样过时——仅覆盖 3 种基本模式，未提及 EASTMONEY_DKBAR
667 9:49p ✅ README.md 已更新：修正"Eastmoney-style"误导性表述，明确项目目标为单股交易验证
668 " ✅ docs/indicator_formula.md 已更新：移除"approximates Eastmoney"的误导性声明，新增 eastmoney_dkbar 章节
669 9:50p ✅ docs/plan-05-15(2).md 已完全重写：从"DK Trend 复刻计划"转为"单股量化交易计划"
670 " ✅ src/indicators/dktrend.py 模块文档字符串和注释已更新以反映新目标
671 " ✅ calibrate_eastmoney_dkbar.py 和 plot_dktrend.py 的文档字符串和图表标题已更新
672 " ✅ DK Bar 配置文件（eastmoney_dkbar_test.yaml、wfo_eastmoney_dkbar.yaml）添加研究性质声明头注释
S57 完成项目目标表述和文档措辞的系统化更新：将 single-stock-quant 从"东财指标复刻"重新定位为"单股量化交易系统"，涵盖 10 个文件的修改 (May 17, 9:51 PM)
S58 session 完成了两阶段工作：(1) 全面分析 single-stock-quant 项目的问题和改进方向；(2) 系统化更新项目文档将定位从"东财指标复刻"改为"单股量化交易系统" (May 17, 9:52 PM)
S59 用户请求Codex重构最新的plan文件，目标单一：提升单股量化收益 (May 17, 9:53 PM)
S60 用户请求Codex重构最新的plan文件（docs/plan-05-15(2).md），目标单一：提升单股量化收益 (May 17, 10:00 PM)
S61 用户请求Codex重构最新的plan文件（docs/plan-05-15(2).md），目标单一：提升单股量化收益 (May 17, 10:01 PM)
673 10:02p 🔵 000783 DK Bar参数网格搜索结果：最优配置为sma205_wma5组合
674 " 🔵 000783 WFO两折OOS结果：跨折Sharpe波动极大，Bootstrap置信区间极宽
675 " 🔵 数据输出目录包含丰富的跨股票WFO和多模式回测结果
676 " 🔄 plan-05-15(2).md 全面重构为单一目标：提升000783单股量化收益
S62 重构最新plan文件 docs/plan-05-15(2).md，目标单一：提升单股量化收益 (May 17, 10:03 PM)
S63 用户询问"当前模型公式是什么"——Codex深入解读了项目核心模型实现，给出了完整的公式、算法和架构说明 (May 17, 10:04 PM)
677 10:04p 🔵 dktrend.py是核心指标模块，支持8种趋势模式，DK Bar参数已注明不追求官方复刻
678 10:05p 🔵 single_stock.py回测引擎支持10种退出机制和5层仓位管理
679 " 🔵 WFO复合目标函数以Calmar为主权重（40%），硬约束限制年交易4-24次且回撤≤40%
680 " 🔵 compute_signal_quality评分函数为BUY信号提供8维度0-100质量分
S64 Complete plan-05-15(2).md — execute all five R01-R05 tasks for 000783 single-stock return improvement, including P0 refactoring of the scoring pipeline from anchor-matching to pure return-quality optimization (May 17, 10:07 PM)
681 10:13p 🔵 Strategic pivot: single-stock quant project shifts from anchor-matching to pure return optimization
682 " 🔵 000783 return grid shows sma205/wma5/c3 as current WFO baseline with low drawdown, sma60 variants as higher-return alternatives
683 " 🔵 Codebase structure: single-stock-quant project has mature WFO framework, trade forensics, and experiment infrastructure
684 10:16p 🔵 WFO JSON output files contain widespread NaN values across heatmap and stability fields
685 " 🟣 R01 implemented: return_quality_score extracted, trade concentration metrics added, anchor diagnostics removed from primary scoring
686 10:17p 🟣 WFO now outputs per-fold trade concentration metrics and JSON-safe NaN/inf sanitization
687 10:18p 🟣 New orchestration script run_000783_return_plan.py implements all five R01-R05 tasks from plan-05-15(2)
688 10:19p 🟣 All 16 existing tests pass after P0/P1 code changes; plan execution script launched successfully
689 " 🔵 Plan execution script still running after 30s — 252 grid re-enrichment + 34 WFO runs expected to take minutes
690 10:20p 🔵 run_000783_return_plan.py crashes at YAML write: numpy int64 not serializable by PyYAML safe_dump
691 " 🔴 YAML serialization crash fixed: _native() function converts numpy scalars to Python native types before safe_dump
692 " 🔵 Re-launched plan script still computing after 30s — full workload of ~300 backtests in progress
693 10:21p 🟣 Plan-05-15(2) fully executed: all five R01-R05 outputs produced with best strategy identified
694 " ⚖️ Exit-layer optimization confirmed as primary return lever over entry signal tuning
695 10:22p ✅ Plan-05-15(2).md updated with completion status and execution results section 7.1
696 " 🟣 Plan-05-15(2) fully completed: P0 refactoring + R01-R05 execution delivered in single session
697 10:23p ✅ Code quality: ruff linting errors resolved, unused variable removed from nested WFO path
698 " ✅ Final quality gate: all 201 tests pass, all ruff linting passes, clean git diff shows targeted changes only
S65 Complete plan-05-15(2).md — execute all five R01-R05 tasks: refactor scoring from anchor-matching to pure return-quality, re-rank grid candidates, run fixed-parameter WFO on top-10, test exit/entry experiment overlays, and produce best strategy config (May 17, 10:24 PM)
699 10:25p 🔵 Best experiment trade-level details confirmed: 7 trades across 3 OOS folds, all exits via intrapos_dd_stop or signal
### May 18, 2026
700 4:26p 🔵 量化交易策略优化任务启动：多股票最佳单股策略配置分析
701 4:27p 🔵 项目结构探索：single-stock-quant 量化交易系统的关键配置与数据文件
702 4:28p 🔵 全面代码审查完成：single-stock-quant 项目架构与多股票策略配置任务的理解
703 4:29p 🔵 WFO 结果跨目录对比：5只股票在不同配置下的参数表现
704 4:30p 🟣 新增多股票最佳策略配置导出脚本 export_selected_single_stock_best.py
705 " 🟣 5只股票最佳单股策略配置产出完成：每只独立 MACD 参数 + 完整交易明细
706 4:31p 🟣 任务完成：多股票最佳单股策略配置导出脚本最终版完善
S66 多股票最佳单股策略配置导出：产出 5 只 WFO passing 股票的独立最佳参数配置及 246 笔逐笔交易明细 (May 18, 4:32 PM)
**Investigated**: 探索了 single-stock-quant 项目的完整架构，包括：
    1. 核心回测引擎 src/backtest/single_stock.py 的 run_single_stock_backtest() 函数（支持 T+1 开盘执行、10种退出机制、信号过滤、trade_log 含 14 个归因字段）
    2. 统一参数构建函数 src/backtest/config.py 的 build_bt_kwargs()
    3. 趋势指标系统 src/indicators/dktrend.py（9 种模式：MACD_CROSS、MA_CROSS、BOLL_TREND、DONCHIAN_BREAKOUT 等）
    4. 多个配置文件：prod-v1.yaml（5 股 WFO 验证）、s_final.yaml（SE1-SE6 实验结论）、wfo_stable.yaml（固定 exit 仅优化 MACD）
    5. E_SINGLE_stable 和 E_SINGLE_final 实验的 WFO 结果对比（发现 E_SINGLE_final 参数更多反而在 300750 上表现更差）
    6. 现有 000783 单股分析流程（run_000783_return_plan.py）作为参考模式
    7. 项目 plan-05-15(2).md 定义的 P3 阶段目标：每只股票独立选参

**Learned**: 1. E_SINGLE_stable（仅优化 MACD fast/signal）的 combined Sharpe 更稳定，E_SINGLE_final（5 参数网格）扩大参数反而引入过拟合：300750 从 stable 的 0.537 跌到 final 的 0.413
    2. 300059 在所有 WFO 配置下 3 个 fold 参数都不一致（10→12→14），提示该股趋势特征不稳定
    3. 所有股票几乎全以 profit_lock 退出（占比 >90%），说明盈利保护卖出机制起主导作用
    4. 交易明细 trade_log 包含 mae/mfe 字段，可分析每笔交易的最大浮亏和最大浮盈
    5. 601318（中国平安）全样本收益为 -3.39%，尽管 WFO OOS 收益 24.06% 为正，应标记为"观察"标的
    6. 项目使用真实 A 股成本模型：commission 2.5bps ×2 + slippage 2bps ×2 + stamp_duty 5bps = 合计约 12bps 每笔往返

**Completed**: 1. 创建并完善了 scripts/export_selected_single_stock_best.py（~180行），功能包括：
       - 从 WFO JSON 的 platform_by_fold 中提取每只股票最新 fold 的最佳 MACD 参数
       - 对每只股票执行全样本回测并输出完整 trade_log
       - 生成多种格式输出：YAML 配置、汇总 CSV、合并交易明细 CSV、单股交易 CSV
    2. 修复了 _native() 函数中 pandas NaN 序列化的语法错误
    3. 新增 --config-output 参数以同时写入 configs/research/ 目录
    4. 产出文件清单：
       - configs/research/selected_single_stock_best.yaml（完整配置）
       - data/output/selected_single_stock_best/best_configs.yaml
       - data/output/selected_single_stock_best/best_configs_summary.csv（5 行汇总）
       - data/output/selected_single_stock_best/trade_details.csv（246 行合并交易）
       - data/output/selected_single_stock_best/{symbol}_trades.csv（5 个单股交易文件）
    5. ruff lint 通过，零警告零错误
    6. 5 只股票结果汇总：
       - 002475 立讯精密 MACD(14,26,8) 全样本回报 45.07% 年化 6.30%
       - 300059 东方财富 MACD(14,26,8) 全样本回报 15.55% 年化 2.40%
       - 300750 宁德时代 MACD(10,26,8) 全样本回报 139.30% 年化 15.35%
       - 600030 中信证券 MACD(12,26,10) 全样本回报 43.86% 年化 6.18%
       - 601318 中国平安 MACD(10,26,8) 全样本回报 -3.39% 年化 -0.56%

**Next Steps**: 当前任务已完成交付。用户请求的多股票最佳单股策略配置和逐笔交易明细已全部产出。
    工作区存在原有的未处理变更：AGENTS.md 修改和 docs/000783.md 删除，这些不属于本次任务范围。


Access 1646k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>