#!/usr/bin/env python
"""Read current_positions.json and send rotation trade notification via WeCom webhook."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.settings import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Send rotation trade notification")
    parser.add_argument("--state-file", required=True, help="Path to current_positions.json")
    parser.add_argument("--config", help="Config file path")
    args = parser.parse_args()

    state_path = Path(args.state_file)
    if not state_path.exists():
        print(f"State file not found: {state_path}")
        return 1

    state = json.loads(state_path.read_text(encoding="utf-8"))
    cfg = load_config(args.config)
    webhook_url = cfg.get("notify", {}).get("wecom_webhook_url", "")
    if not webhook_url:
        print("未配置 WeCom Webhook URL，跳过通知")
        return 0

    from src.notify import WecomWebhookHandler

    handler = WecomWebhookHandler(webhook_url)
    positions = state.get("symbols", [])
    date_str = state.get("date", "unknown")
    ann_ret = state.get("annualized_return", 0.0)
    calmar = state.get("calmar_ratio", 0.0)

    msg = (
        f"【轮动持仓】{date_str}\n"
        f"当前持仓: {', '.join(positions) if positions else '空仓'}\n"
        f"年化收益: {ann_ret*100:.1f}% | Calmar: {calmar:.2f}"
    )
    handler.send_markdown(msg)
    print(f"已发送轮动通知: {len(positions)} 只持仓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
