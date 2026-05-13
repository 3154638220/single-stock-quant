from pathlib import Path

from src.data_fetcher.db_manager import DuckDBManager


def test_duckdb_manager_does_not_apply_legacy_migrations_by_default(tmp_path: Path):
    db_path = tmp_path / "market.duckdb"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
paths:
  duckdb_path: {db_path}
database:
  auto_backfill_derived_on_init: false
  apply_legacy_migrations: false
""",
        encoding="utf-8",
    )

    with DuckDBManager(config_path=cfg_path) as db:
        tables = {
            str(row[0])
            for row in db.connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }

    assert "a_share_daily" in tables
    assert "data_fetch_audit" in tables
    assert "schema_migrations" not in tables
    assert "oos_tracking" not in tables
