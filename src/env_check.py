"""Environment checks for the single-stock trend system."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

from src.data_fetcher.akshare_client import fetch_a_share_daily
from src.settings import load_config, project_root


def _ok(msg: str, *, quiet: bool) -> None:
    if not quiet:
        print(f"[OK] {msg}", flush=True)


def _fail(msg: str, *, quiet: bool) -> None:
    if not quiet:
        print(f"[FAIL] {msg}", flush=True)


def _dns_summary(hosts: list[str]) -> str:
    pairs: list[str] = []
    for host in hosts:
        try:
            infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            ips: list[str] = []
            for item in infos:
                ip = str(item[4][0])
                if ip not in ips:
                    ips.append(ip)
            pairs.append(f"{host}={'/'.join(ips[:2])}")
        except Exception as exc:
            pairs.append(f"{host}=ERR:{type(exc).__name__}")
    return "; ".join(pairs)


def run_checks(*, config: Path | None, quiet: bool) -> int:
    """运行所有环境检查，返回失败数（0=全部通过）。"""
    root = project_root()
    cfg = load_config(config)
    paths = cfg.get("paths", {})
    duckdb_rel = paths.get("duckdb_path", "data/market.duckdb")
    duckdb_path = Path(duckdb_rel)
    if not duckdb_path.is_absolute():
        duckdb_path = root / duckdb_path

    failed = 0

    # Python 3.10+
    vi = sys.version_info
    if vi.major != 3 or vi.minor < 10:
        _fail(f"期望 Python >=3.10，当前 {vi.major}.{vi.minor}.{vi.micro}", quiet=quiet)
        failed += 1
    else:
        _ok(f"Python {vi.major}.{vi.minor}.{vi.micro}", quiet=quiet)

    # Core import checks
    for module in ["pandas", "numpy", "yaml", "requests", "duckdb"]:
        try:
            __import__(module)
            _ok(f"import {module}", quiet=quiet)
        except Exception as e:
            _fail(f"import {module} failed: {e}", quiet=quiet)
            failed += 1

    # DuckDB 可写
    try:
        import duckdb

        duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(duckdb_path))
        try:
            con.execute("CREATE TABLE IF NOT EXISTS _env_probe (x INTEGER)")
            con.execute("INSERT INTO _env_probe VALUES (1)")
            con.execute("DELETE FROM _env_probe WHERE x = 1")
            con.execute("DROP TABLE IF EXISTS _env_probe")
        finally:
            con.close()
        _ok(f"DuckDB 可写: {duckdb_path}", quiet=quiet)
    except Exception as e:
        _fail(f"DuckDB 不可写或无法打开 {duckdb_path}: {e}", quiet=quiet)
        failed += 1

    # AkShare 连通
    try:
        df = fetch_a_share_daily(
            "000001",
            "20240102",
            "20240110",
            adjust=cfg.get("akshare", {}).get("adjust", "qfq"),
            timeout_sec=float(cfg.get("akshare", {}).get("request_timeout_sec", 10.0)),
        )
        if df is None or df.empty:
            _fail("AkShare 返回空表（网络或接口异常）", quiet=quiet)
            failed += 1
        else:
            _ok(f"AkShare 连通（样本: 000001 日线 {len(df)} 行）", quiet=quiet)
    except Exception as e:
        dns_msg = _dns_summary(
            ["finance.sina.com.cn", "stock.finance.sina.com.cn", "push2his.eastmoney.com"]
        )
        _fail(f"AkShare 连通性失败: {e} | DNS: {dns_msg}", quiet=quiet)
        failed += 1

    # Conda 环境提示
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env == "quant-system":
        _ok("CONDA_DEFAULT_ENV=quant-system", quiet=quiet)
    elif not quiet:
        print(
            f"[WARN] 当前 conda 环境为 {conda_env!r}，建议使用项目依赖环境",
            flush=True,
        )

    return 1 if failed else 0
