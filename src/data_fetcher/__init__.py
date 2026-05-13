from .akshare_client import fetch_a_share_daily, fill_derived_daily_fields, list_default_universe_symbols

__all__ = [
    "fetch_a_share_daily",
    "fill_derived_daily_fields",
    "list_default_universe_symbols",
    "DuckDBManager",
    "SymbolUpdateResult",
    "QualityConfig",
    "QualityReport",
    "check_split_jump",
    "run_quality_checks",
    "validate_daily_frame",
    "IndexFetchSpec",
    "DEFAULT_INDEX_SPECS",
    "parse_index_specs",
    "standardize_index_daily",
    "load_stock_name_cache",
    "load_stock_name_map",
    "resolve_stock_name_cache_path",
    "resolve_stock_names",
    "attach_stock_names",
]


def __getattr__(name: str):
    if name in {"QualityConfig", "QualityReport", "check_split_jump", "run_quality_checks", "validate_daily_frame"}:
        from .data_quality import (
            QualityConfig,
            QualityReport,
            check_split_jump,
            run_quality_checks,
            validate_daily_frame,
        )

        return {
            "QualityConfig": QualityConfig,
            "QualityReport": QualityReport,
            "check_split_jump": check_split_jump,
            "run_quality_checks": run_quality_checks,
            "validate_daily_frame": validate_daily_frame,
        }[name]
    if name in {"DuckDBManager", "SymbolUpdateResult"}:
        from .db_manager import DuckDBManager, SymbolUpdateResult

        return {"DuckDBManager": DuckDBManager, "SymbolUpdateResult": SymbolUpdateResult}[name]
    if name in {
        "IndexFetchSpec",
        "DEFAULT_INDEX_SPECS",
        "parse_index_specs",
        "standardize_index_daily",
    }:
        from . import index_benchmarks

        return getattr(index_benchmarks, name)
    if name in {
        "load_stock_name_cache",
        "load_stock_name_map",
        "resolve_stock_name_cache_path",
        "resolve_stock_names",
        "attach_stock_names",
    }:
        from . import stock_name_cache

        return getattr(stock_name_cache, name)
    raise AttributeError(name)
