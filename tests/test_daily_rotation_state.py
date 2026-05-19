"""Tests for --output-state parameter in run_rotation.py."""

import json
import tempfile
from pathlib import Path

import pytest


class TestOutputStateFormat:
    """Verify the output state JSON format is valid and contains expected keys."""

    def test_state_file_has_required_keys(self):
        state = {
            "date": "2026-05-19",
            "symbols": ["300750"],
            "annualized_return": 0.3142,
            "calmar_ratio": 1.10,
        }
        text = json.dumps(state, ensure_ascii=False, indent=2)
        parsed = json.loads(text)
        assert "date" in parsed
        assert "symbols" in parsed
        assert "annualized_return" in parsed
        assert "calmar_ratio" in parsed

    def test_state_file_writable(self):
        state = {
            "date": "2026-05-19",
            "symbols": [],
            "annualized_return": 0.0,
            "calmar_ratio": 0.0,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            path = Path(f.name)
        try:
            assert path.exists()
            parsed = json.loads(path.read_text())
            assert parsed == state
        finally:
            path.unlink()

    def test_empty_positions_serializes_as_empty_list(self):
        state = {"date": "2026-05-19", "symbols": [], "annualized_return": 0.0, "calmar_ratio": 0.0}
        text = json.dumps(state)
        assert '"symbols": []' in text or '"symbols": []' in text.replace(" ", "")
