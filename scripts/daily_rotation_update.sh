#!/bin/bash
# daily_rotation_update.sh — X3 轮动策略的每日执行流
# crontab: 30 16 * * 1-5 /path/to/scripts/daily_rotation_update.sh >> /var/log/quant/daily_rotation.log 2>&1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WATCHLIST="$ROOT/configs/watchlist_x3.txt"
SYMBOL_PARAMS="$ROOT/configs/symbol_params.yaml"
SECTOR_MAP="$ROOT/configs/sector_map.yaml"
STATE_DIR="$ROOT/data/rotation_state"
STATE_FILE="$STATE_DIR/current_positions.json"
CONFIG="$ROOT/configs/prod-v1.yaml"
LOG_DIR="$ROOT/data/logs"
mkdir -p "$STATE_DIR" "$LOG_DIR"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') daily_rotation_update start ==="

# 1. 更新 X3 池数据
echo "[1/3] 更新日线数据..."
while IFS= read -r sym; do
  [ -z "$sym" ] || [[ "$sym" == \#* ]] && continue
  python "$ROOT/scripts/fetch_stock.py" \
    --symbol "$sym" \
    --check-quality --fail-on-quality \
    --quality-min-rows 5 \
    --config "$CONFIG" 2>&1 || echo "WARN: $sym 数据更新失败"
  sleep 2
done < "$WATCHLIST"

# 2. 计算当日轮动信号
echo "[2/3] 计算轮动信号..."
python "$ROOT/scripts/run_rotation.py" \
  --watchlist "$WATCHLIST" \
  --symbol-params "$SYMBOL_PARAMS" \
  --ranking-mode trend_strength \
  --top-n 1 --rebalance-freq 10 \
  --market-regime-mode exit \
  --regime-fast-ma-period 60 \
  --regime-ma-period 120 \
  --regime-drawdown-trigger 0.15 \
  --config "$CONFIG" \
  --export-results \
  --output-state "$STATE_FILE" 2>&1

# 3. 发送轮动通知
echo "[3/3] 检查是否需要发送通知..."
if [ -f "$STATE_FILE" ]; then
  python "$ROOT/scripts/notify_rotation.py" \
    --state-file "$STATE_FILE" \
    --config "$CONFIG" 2>&1 || echo "WARN: 通知发送失败"
else
  echo "WARN: 未生成持仓状态文件，跳过通知"
fi

echo "=== $(date '+%Y-%m-%d %H:%M:%S') daily_rotation_update done ==="
