#!/bin/bash
# Daily single-stock signal pipeline (crontab: 30 16 * * 1-5)
# Usage: bash scripts/daily_single_stock.sh [config]
#
# 1. Update market data (stock + CSI300 index)
# 2. Run signal detection for watchlist stocks
# 3. Send WeCom notification on actionable signals

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-configs/single_stock/300750_best.yaml}"
WATCHLIST="$ROOT/configs/watchlist_single.txt"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') Single-stock daily pipeline ==="

# Step 1: Fetch latest data (append mode) — stocks + CSI300
echo "[1/3] Fetching stock data..."
python "$ROOT/scripts/fetch_stock.py" \
  --symbols "$(tr '\n' ' ' < "$WATCHLIST")" \
  --config "$CONFIG" \
  --mode latest || echo "WARNING: fetch may have partial failures"

echo "[1/3] Fetching CSI300 index data..."
python "$ROOT/scripts/fetch_index.py" \
  --symbol 000300 \
  --config "$CONFIG" || echo "WARNING: CSI300 fetch may have failed"

# Step 2: Run signals
echo "[2/3] Computing signals..."
python "$ROOT/scripts/run_signal.py" \
  --watchlist "$(tr '\n' ' ' < "$WATCHLIST")" \
  --config "$CONFIG"

# Step 3: Send WeCom notification for actionable signals
echo "[3/3] Sending notifications..."
for sym in $(tr '\n' ' ' < "$WATCHLIST"); do
  python "$ROOT/scripts/notify_single_stock.py" \
    --symbol "$sym" \
    --config "$CONFIG" || echo "WARNING: notification failed for $sym"
done

echo "Pipeline complete at $(date '+%Y-%m-%d %H:%M:%S')"
