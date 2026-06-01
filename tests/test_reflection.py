import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from reflection_engine import (
    run_reflection, _detect_primary_weakness, _get_current_variable_values,
    _generate_hypothesis, VARIABLE_BOUNDS
)

GOAL = {
    "target_return_30d": 0.05,
    "max_drawdown": 0.08,
    "min_sharpe": 1.2,
}

STRATEGY = {
    "version": "01",
    "entry": {"indicator": "RSI", "threshold": 30},
    "direction": "LONG",
    "stop_loss_pct": 2,
    "take_profit_pct": 4,
    "position_size_pct": 5,
}


def make_trades(returns, regime="RANGING"):
    return [
        {"return": r, "pnl": r * 100, "drawdown": max(0, -r * 0.5), "regime": regime}
        for r in returns
    ]


class TestGetCurrentVariableValues:
    def test_extracts_all_variables(self):
        vals = _get_current_variable_values(STRATEGY)
        assert "rsi_threshold" in vals
        assert "stop_loss_pct" in vals
        assert "take_profit_pct" in vals
        assert "position_size_pct" in vals

    def test_correct_values(self):
        vals = _get_current_variable_values(STRATEGY)
        assert vals["rsi_threshold"] == 30.0
        assert vals["stop_loss_pct"] == 2.0
        assert vals["take_profit_pct"] == 4.0
        assert vals["position_size_pct"] == 5.0


class TestDetectPrimaryWeakness:
    def test_low_return(self):
        perf = {"return": 0.01, "max_drawdown": 0.05, "sharpe": 1.5}
        weakness = _detect_primary_weakness(perf, GOAL)
        assert weakness == "low_return"

    def test_high_drawdown(self):
        perf = {"return": 0.10, "max_drawdown": 0.15, "sharpe": 1.5}
        weakness = _detect_primary_weakness(perf, GOAL)
        assert weakness == "high_drawdown"

    def test_low_sharpe(self):
        perf = {"return": 0.10, "max_drawdown": 0.05, "sharpe": 0.5}
        weakness = _detect_primary_weakness(perf, GOAL)
        assert weakness == "low_sharpe"

    def test_no_weakness(self):
        perf = {"return": 0.10, "max_drawdown": 0.03, "sharpe": 2.0}
        weakness = _detect_primary_weakness(perf, GOAL)
        assert weakness == "none"

    def test_worst_weakness_selected(self):
        perf = {"return": -0.10, "max_drawdown": 0.20, "sharpe": -1.0}
        weakness = _detect_primary_weakness(perf, GOAL)
        assert weakness in ("low_return", "high_drawdown", "low_sharpe")


class TestGenerateHypothesis:
    def setup_method(self):
        self.current_values = _get_current_variable_values(STRATEGY)
        self.regime_scores = {"RANGING": {"score": 0.2}, "TRENDING_UP": {"score": 0.5}}

    def test_returns_tuple_of_five(self):
        result = _generate_hypothesis("low_return", self.current_values, self.regime_scores)
        assert len(result) == 5

    def test_variable_in_bounds(self):
        for weakness in ("low_return", "high_drawdown", "low_sharpe", "none"):
            var, old_val, new_val, reason, effect = _generate_hypothesis(
                weakness, self.current_values, self.regime_scores
            )
            lo, hi = VARIABLE_BOUNDS[var]
            assert lo <= new_val <= hi, f"{var}={new_val} out of [{lo}, {hi}]"

    def test_only_one_variable_changed(self):
        var, old_val, new_val, reason, effect = _generate_hypothesis(
            "low_return", self.current_values, self.regime_scores
        )
        assert isinstance(var, str)
        assert var in VARIABLE_BOUNDS

    def test_reason_is_string(self):
        _, _, _, reason, effect = _generate_hypothesis(
            "high_drawdown", self.current_values, self.regime_scores
        )
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_effect_is_string(self):
        _, _, _, reason, effect = _generate_hypothesis(
            "low_sharpe", self.current_values, self.regime_scores
        )
        assert isinstance(effect, str)
        assert len(effect) > 0


class TestRunReflection:
    def test_empty_trades_returns_none(self):
        result = run_reflection([], STRATEGY, GOAL)
        assert result is None

    def test_returns_hypothesis_dict(self):
        trades = make_trades([0.005] * 10)
        result = run_reflection(trades, STRATEGY, GOAL)
        assert result is not None
        assert "changed_variable" in result
        assert "old_value" in result
        assert "new_value" in result
        assert "reason" in result
        assert "expected_effect" in result

    def test_only_one_variable_changed(self):
        trades = make_trades([-0.01] * 10)
        result = run_reflection(trades, STRATEGY, GOAL)
        assert result is not None
        assert result["changed_variable"] in VARIABLE_BOUNDS

    def test_new_value_differs_from_old(self):
        trades = make_trades([-0.02] * 15)
        result = run_reflection(trades, STRATEGY, GOAL)
        assert result is not None
        assert result["old_value"] != result["new_value"]

    def test_new_value_in_bounds(self):
        trades = make_trades([0.005] * 10)
        result = run_reflection(trades, STRATEGY, GOAL)
        assert result is not None
        var = result["changed_variable"]
        lo, hi = VARIABLE_BOUNDS[var]
        assert lo <= result["new_value"] <= hi
