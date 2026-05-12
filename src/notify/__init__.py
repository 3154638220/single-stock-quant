"""外部通知渠道集成：企业微信 Webhook、钉钉等。

P2-8: 提供告警回调 handler 示例实现，供 ICMonitor 等组件注册使用。
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import requests

from src.signals.types import Signal

logger = logging.getLogger(__name__)


class WecomWebhookHandler:
    """P2-8: 企业微信机器人 Webhook 告警处理器。

    Send markdown or text messages through a WeCom group robot.
    """

    def __init__(self, url: str, *, timeout: float = 10.0, mention_all: bool = False) -> None:
        """
        Parameters
        ----------
        url : str
            企业微信群机器人 Webhook 地址。
        timeout : float
            HTTP 请求超时秒数。
        mention_all : bool
            是否 @所有人。
        """
        self.url = str(url)
        self.timeout = float(timeout)
        self.mention_all = bool(mention_all)

    def __call__(self, alert: object) -> bool:
        """发送单条告警消息到企业微信。

        Parameters
        ----------
        alert : ICDecayAlert 或任何实现了 __str__ 的对象

        Returns
        -------
        bool : 发送是否成功
        """
        message = str(alert)
        return self.send_markdown(message)

    def send_markdown(self, content: str) -> bool:
        """以 Markdown 格式发送消息。

        Parameters
        ----------
        content : str
            Markdown 格式的消息正文。

        Returns
        -------
        bool
        """
        payload: dict = {
            "msgtype": "markdown",
            "markdown": {
                "content": content,
            },
        }
        if self.mention_all:
            payload["markdown"]["mentioned_list"] = ["@all"]

        try:
            resp = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("errcode") != 0:
                logger.warning(
                    "企业微信 Webhook 返回错误: errcode=%s, errmsg=%s",
                    result.get("errcode"),
                    result.get("errmsg", ""),
                )
                return False
            return True
        except requests.exceptions.RequestException as exc:
            logger.error("企业微信 Webhook 请求失败: %s", exc)
            return False
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("企业微信 Webhook 响应解析失败: %s", exc)
            return False

    def send_text(self, content: str, mentioned_list: Optional[list[str]] = None) -> bool:
        """以纯文本格式发送消息。

        Parameters
        ----------
        content : str
            消息正文（最长 2048 字节）。
        mentioned_list : list[str] or None
            要 @的成员 userid 列表；["@all"] 表示 @所有人。

        Returns
        -------
        bool
        """
        payload: dict = {
            "msgtype": "text",
            "text": {
                "content": content,
            },
        }
        if mentioned_list:
            payload["text"]["mentioned_list"] = mentioned_list

        try:
            resp = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("errcode") != 0:
                logger.warning(
                    "企业微信 Webhook 返回错误: errcode=%s, errmsg=%s",
                    result.get("errcode"),
                    result.get("errmsg", ""),
                )
                return False
            return True
        except requests.exceptions.RequestException as exc:
            logger.error("企业微信 Webhook 请求失败: %s", exc)
            return False
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("企业微信 Webhook 响应解析失败: %s", exc)
            return False


def send_trend_signal(
    handler: WecomWebhookHandler,
    symbol: str,
    stock_name: str,
    signal: Signal,
    close: float,
    dk_run_len: int,
    trade_date: str,
) -> bool:
    """Send a DK trend signal message through a WeCom webhook handler."""
    action = "买入信号" if signal == Signal.BUY else "卖出信号"
    trend = "趋势刚变红" if signal == Signal.BUY else "趋势刚变绿"
    content = (
        f"【多空趋势信号】{stock_name} ({symbol})\n"
        f"{action} | {trade_date}\n"
        f"收盘价：{close:.2f} | {trend} | 连续 {int(dk_run_len)} 天"
    )
    return handler.send_markdown(content)
