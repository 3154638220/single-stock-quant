#!/bin/bash
# quarterly_retrain.sh — 每季度末运行 WFO 重训练，刷新参数和 watchlist
# 建议 crontab: 0 10 1 1,4,7,10 * /path/to/scripts/quarterly_retrain.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="$ROOT/configs/prod-v1.yaml"
WATCHLIST="$ROOT/configs/watchlist_wfo_passing.txt"
DATE_TAG=$(date +%Y%m)
OUT_DIR="$ROOT/data/output/experiments/WFO_${DATE_TAG}_refresh"
mkdir -p "$OUT_DIR"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') quarterly_retrain start ==="

# Copy DuckDB for read-only WFO access
cp "$ROOT/data/market.duckdb" /tmp/market_retrain.duckdb

while IFS= read -r sym; do
  [ -z "$sym" ] && continue
  echo "WFO retrain: $sym"
  python "$ROOT/scripts/run_wfo.py" \
    --symbol "$sym" \
    --config "$CONFIG" \
    --start 2020-01-01 \
    --train-days 756 --oos-days 252 \
    --duckdb-path /tmp/market_retrain.duckdb \
    --export-results 2>&1
  # Move result to refresh dir
  mv "$ROOT/data/output/${sym}_wfo_${DATE_TAG}"*.json "$OUT_DIR/" 2>/dev/null || true
done < "$WATCHLIST"

rm -f /tmp/market_retrain.duckdb

# Evaluate results
echo ""
echo "=== OOS Sharpe summary ==="
python -c "
import json, glob
for path in sorted(glob.glob('$OUT_DIR/*_wfo_*.json')):
    with open(path) as f:
        d = json.load(f)
    sharpe = d['aggregated'].get('sharpe_ratio_combined', float('nan'))
    sym = d['symbol']
    flag = '⚠️  <0.15 — GREYLIST' if sharpe < 0.15 else 'OK'
    print(f'  {sym}: OOS Sharpe={sharpe:.3f}  {flag}')
"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') quarterly_retrain done ==="
