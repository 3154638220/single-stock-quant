from pathlib import Path

from src.data_fetcher.stock_name_cache import (
    load_stock_name_cache,
    resolve_stock_name_cache_path,
    resolve_stock_names,
)


def test_load_stock_name_cache_normalizes_common_columns(tmp_path: Path):
    path = tmp_path / "names.csv"
    path.write_text("证券代码,证券简称\n600000,浦发银行\n1,平安银行\n", encoding="utf-8")

    names = load_stock_name_cache(path)

    assert names.to_dict("records") == [
        {"symbol": "600000", "name": "浦发银行"},
        {"symbol": "000001", "name": "平安银行"},
    ]


def test_resolve_stock_names_falls_back_to_symbol(tmp_path: Path):
    path = tmp_path / "names.csv"
    path.write_text("symbol,name\n600000,浦发银行\n", encoding="utf-8")

    resolved = resolve_stock_names(["600000", "000001"], path)

    assert resolved["600000"] == "浦发银行"
    assert resolved["000001"] == "000001"


def test_resolve_stock_name_cache_path_uses_project_relative_default():
    path = resolve_stock_name_cache_path({"paths": {"stock_name_cache": "data/names.csv"}})

    assert path.name == "names.csv"
    assert path.is_absolute()
