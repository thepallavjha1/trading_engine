import sys
import shutil
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import yaml


STRATEGY = {
    "version": "01",
    "entry": {"indicator": "RSI", "threshold": 30},
    "direction": "LONG",
    "stop_loss_pct": 2,
    "take_profit_pct": 4,
    "position_size_pct": 5,
    "timeframe": "1h",
    "lookback_candles": 200,
}


@pytest.fixture(autouse=True)
def patch_paths(tmp_path, monkeypatch):
    history_dir = tmp_path / "history" / "strategy_versions"
    history_dir.mkdir(parents=True)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    strategy_file = configs_dir / "strategy.yaml"
    with open(strategy_file, "w") as f:
        yaml.dump(STRATEGY, f)

    reflections_file = data_dir / "reflections.jsonl"
    reflections_file.touch()

    import utils
    monkeypatch.setattr(utils, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(utils, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(utils, "REFLECTIONS_FILE", reflections_file)

    import version_manager
    monkeypatch.setattr(version_manager, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(version_manager, "STRATEGY_FILE", strategy_file)
    monkeypatch.setattr(version_manager, "REFLECTIONS_FILE", reflections_file)

    yield


class TestSaveVersionSnapshot:
    def test_creates_file(self, tmp_path):
        from version_manager import save_version_snapshot
        strategy = dict(STRATEGY)
        path = save_version_snapshot(strategy)
        assert path.exists()

    def test_file_contains_version(self, tmp_path):
        from version_manager import save_version_snapshot
        strategy = dict(STRATEGY)
        path = save_version_snapshot(strategy)
        loaded = yaml.safe_load(path.read_text())
        assert loaded["version"] == "01"


class TestBumpVersion:
    def test_increments_version(self):
        from version_manager import bump_version
        strategy = dict(STRATEGY)
        new_strat, new_ver = bump_version(strategy)
        assert new_ver == "02"
        assert new_strat["version"] == "02"

    def test_original_unchanged(self):
        from version_manager import bump_version
        strategy = dict(STRATEGY)
        bump_version(strategy)
        assert strategy["version"] == "01"

    def test_increments_from_high(self):
        from version_manager import bump_version
        strategy = dict(STRATEGY, version="09")
        _, new_ver = bump_version(strategy)
        assert new_ver == "10"


class TestApplyModification:
    def test_changes_rsi_threshold(self):
        from version_manager import apply_modification
        result = apply_modification("rsi_threshold", 30, 25, "Test reason", "Test effect")
        assert result["entry"]["threshold"] == 25

    def test_changes_stop_loss(self):
        from version_manager import apply_modification
        result = apply_modification("stop_loss_pct", 2, 1.5, "Test", "Test")
        assert result["stop_loss_pct"] == 1.5

    def test_changes_take_profit(self):
        from version_manager import apply_modification
        result = apply_modification("take_profit_pct", 4, 5, "Test", "Test")
        assert result["take_profit_pct"] == 5

    def test_changes_position_size(self):
        from version_manager import apply_modification
        result = apply_modification("position_size_pct", 5, 3, "Test", "Test")
        assert result["position_size_pct"] == 3

    def test_version_bumped(self):
        from version_manager import apply_modification
        result = apply_modification("stop_loss_pct", 2, 1.5, "Test", "Test")
        assert result["version"] == "02"

    def test_snapshot_saved(self, tmp_path):
        import version_manager
        from version_manager import apply_modification
        apply_modification("stop_loss_pct", 2, 1.5, "Test", "Test")
        snapshots = list(version_manager.HISTORY_DIR.glob("v*.yaml"))
        assert len(snapshots) == 1

    def test_reflection_logged(self, tmp_path):
        import json
        import version_manager
        from version_manager import apply_modification
        apply_modification("take_profit_pct", 4, 5, "Test reason", "Test effect")
        records = [
            json.loads(line) for line in version_manager.REFLECTIONS_FILE.read_text().splitlines()
            if line.strip()
        ]
        assert len(records) == 1
        assert records[0]["changed_variable"] == "take_profit_pct"


class TestGetCurrentVersion:
    def test_returns_string(self):
        from version_manager import get_current_version
        ver = get_current_version()
        assert isinstance(ver, str)


class TestLoadAllVersions:
    def test_empty_initially(self):
        from version_manager import load_all_versions
        versions = load_all_versions()
        assert versions == []

    def test_returns_list_after_save(self):
        from version_manager import save_version_snapshot, load_all_versions
        save_version_snapshot(STRATEGY)
        versions = load_all_versions()
        assert len(versions) == 1
