#!/bin/bash
# Daily single-stock signal pipeline (crontab: 30 16 * * 1-5)
# Usage: bash scripts/daily_single_stock.sh [config]
#
# 1. Update market data
# 2. Run signal detection for watchlist stocks
# 3. Send WeCom notification

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-configs/single_stock/300750_best.yaml}"
WATCHLIST="$ROOT/configs/watchlist_single.txt"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') Single-stock daily pipeline ==="

# Step 1: Fetch latest data (append mode)
echo "[1/3] Fetching stock data..."
python "$ROOT/scripts/fetch_stock.py" \
  --symbols "$(tr '\n' ' ' < "$WATCHLIST")" \
  --config "$CONFIG" \
  --mode latest || echo "WARNING: fetch may have partial failures"

# Step 2: Run signals
echo "[2/3] Computing signals..."
python "$ROOT/scripts/run_signal.py" \
  --watchlist "$(tr '\n' ' ' < "$WATCHLIST")" \
  --config "$CONFIG"

# Step 3: Summary
echo "[3/3] Pipeline complete at $(date '+%Y-%m-%d %H:%M:%S')"
