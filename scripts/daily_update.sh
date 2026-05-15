#!/bin/bash
# daily_update.sh — 每日收盘后更新数据、扫描信号、发送通知
# 建议 crontab: 30 16 * * 1-5 /path/to/scripts/daily_update.sh >> /var/log/quant/daily.log 2>&1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
WATCHLIST="$ROOT/configs/watchlist_wfo_passing.txt"
CONFIG="$ROOT/configs/prod-v1.yaml"
LOG_DIR="$ROOT/data/logs"
mkdir -p "$LOG_DIR"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') daily_update start ==="

# 1. 更新数据
echo "[1/3] 更新日线数据..."
while IFS= read -r sym; do
  [ -z "$sym" ] && continue
  python "$ROOT/scripts/fetch_stock.py" \
    --symbol "$sym" \
    --check-quality --fail-on-quality \
    --quality-min-rows 5 \
    --config "$CONFIG" 2>&1 || echo "WARN: $sym 数据更新失败"
  sleep 2  # AkShare 限流保护
done < "$WATCHLIST"

# 2. 扫描信号
echo "[2/3] 扫描买卖信号..."
SIGNAL_FILE="$ROOT/data/output/signals/$(date +%Y%m%d)_buy_signals.csv"
mkdir -p "$(dirname "$SIGNAL_FILE")"
SYMS=$(tr '\n' ' ' < "$WATCHLIST")
python "$ROOT/scripts/run_signal.py" \
  --watchlist $SYMS \
  --filter buy \
  --config "$CONFIG" 2>&1 | tee "$SIGNAL_FILE"

# 3. 发送通知
echo "[3/3] 检查是否需要发送通知..."
python -c "
import sys; sys.path.insert(0, '$ROOT')
import pandas as pd
from src.notify import WecomWebhookHandler
from src.settings import load_config

cfg = load_config('$CONFIG')
webhook_url = cfg.get('notify', {}).get('wecom_webhook_url', '')
if not webhook_url:
    print('未配置 WeCom Webhook URL，跳过通知')
    sys.exit(0)

try:
    sig = pd.read_csv('$SIGNAL_FILE')
    # run_signal.py already outputs buy signals; check if table has data rows
    if len(sig) > 1:  # header + data
        handler = WecomWebhookHandler(webhook_url)
        symbols = sig['symbol'].tolist() if 'symbol' in sig.columns else []
        handler.send_markdown(f'【选股信号】$(date +%Y-%m-%d)\n{len(symbols)} 只标的出现买入信号：{symbols}')
        print(f'已发送通知：{len(symbols)} 只标的')
    else:
        print('今日无买入信号')
except Exception as e:
    print(f'通知发送失败: {e}')
" 2>&1

echo "=== $(date '+%Y-%m-%d %H:%M:%S') daily_update done ==="
