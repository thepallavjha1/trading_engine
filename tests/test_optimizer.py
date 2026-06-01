import sys
import json
import yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


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

GOAL = {
    "asset": "BTC/USDT",
    "initial_balance": 10000,
    "target_return_30d": 0.05,
    "max_drawdown": 0.08,
    "min_sharpe": 1.2,
    "reflection_every": 5,
}

TRADES = [
    {"return": r, "pnl": r * 100, "drawdown": max(0, -r), "regime": "RANGING",
     "strategy_version": "01", "asset": "BTC/USDT"}
    for r in [-0.01, 0.02, -0.015, 0.03, 0.01, -0.02, 0.01, 0.02, -0.01, 0.015]
]


@pytest.fixture()
def patch_env(tmp_path, monkeypatch):
    history_dir = tmp_path / "history" / "strategy_versions"
    history_dir.mkdir(parents=True)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    strategy_file = configs_dir / "strategy.yaml"
    with open(strategy_file, "w") as f:
        yaml.dump(STRATEGY, f)

    goal_file = configs_dir / "goal.yaml"
    with open(goal_file, "w") as f:
        yaml.dump(GOAL, f)

    trades_file = data_dir / "trades.jsonl"
    with open(trades_file, "w") as f:
        for t in TRADES:
            f.write(json.dumps(t) + "\n")

    reflections_file = data_dir / "reflections.jsonl"
    reflections_file.touch()

    import utils
    monkeypatch.setattr(utils, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(utils, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(utils, "TRADES_FILE", trades_file)
    monkeypatch.setattr(utils, "REFLECTIONS_FILE", reflections_file)
    monkeypatch.setattr(utils, "DATA_DIR", data_dir)

    import version_manager
    monkeypatch.setattr(version_manager, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(version_manager, "STRATEGY_FILE", strategy_file)
    monkeypatch.setattr(version_manager, "REFLECTIONS_FILE", reflections_file)

    import trade_logger
    monkeypatch.setattr(trade_logger, "TRADES_FILE", trades_file)

    yield {
        "tmp_path": tmp_path,
        "configs_dir": configs_dir,
        "history_dir": history_dir,
        "trades_file": trades_file,
        "reflections_file": reflections_file,
        "strategy_file": strategy_file,
    }


class TestRunOptimizationCycle:
    def test_returns_dict_with_trades(self, patch_env):
        from optimizer import run_optimization_cycle
        result = run_optimization_cycle(n_recent=10)
        assert result is not None
        assert isinstance(result, dict)

    def test_result_has_required_keys(self, patch_env):
        from optimizer import run_optimization_cycle
        result = run_optimization_cycle(n_recent=10)
        assert result is not None
        assert "changed_variable" in result
        assert "old_value" in result
        assert "new_value" in result
        assert "reason" in result
        assert "expected_effect" in result
        assert "new_version" in result

    def test_strategy_file_updated(self, patch_env):
        from optimizer import run_optimization_cycle
        run_optimization_cycle(n_recent=10)
        strategy = yaml.safe_load(patch_env["strategy_file"].read_text())
        assert strategy["version"] == "02"

    def test_version_snapshot_saved(self, patch_env):
        from optimizer import run_optimization_cycle
        run_optimization_cycle(n_recent=10)
        snapshots = list(patch_env["history_dir"].glob("v*.yaml"))
        assert len(snapshots) >= 1

    def test_reflection_logged(self, patch_env):
        from optimizer import run_optimization_cycle
        run_optimization_cycle(n_recent=10)
        records = [
            json.loads(line)
            for line in patch_env["reflections_file"].read_text().splitlines()
            if line.strip()
        ]
        assert len(records) >= 1

    def test_returns_none_with_insufficient_trades(self, patch_env, monkeypatch):
        import optimizer
        monkeypatch.setattr(optimizer, "load_recent_trades", lambda n: [])
        result = optimizer.run_optimization_cycle(n_recent=10)
        assert result is None

    def test_only_one_variable_changed(self, patch_env):
        from optimizer import run_optimization_cycle
        result = run_optimization_cycle(n_recent=10)
        assert result is not None
        from reflection_engine import VARIABLE_BOUNDS
        assert result["changed_variable"] in VARIABLE_BOUNDS
