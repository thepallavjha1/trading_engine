import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from scoring_engine import (
    compute_sharpe, compute_max_drawdown, compute_win_rate,
    compute_total_return, score, score_by_regime
)

GOAL = {
    "target_return_30d": 0.05,
    "max_drawdown": 0.08,
    "min_sharpe": 1.2,
}


def make_trades(returns, regime="RANGING"):
    return [
        {"return": r, "pnl": r * 100, "drawdown": max(0, -r), "regime": regime}
        for r in returns
    ]


class TestComputeSharpe:
    def test_empty(self):
        assert compute_sharpe([]) == 0.0

    def test_single(self):
        assert compute_sharpe([0.01]) == 0.0

    def test_positive_returns(self):
        import numpy as np
        rng = np.random.default_rng(0)
        returns = [0.01 + rng.normal(0, 0.001) for _ in range(50)]
        sharpe = compute_sharpe(returns)
        assert sharpe > 0

    def test_volatile_returns(self):
        import numpy as np
        returns = list(np.random.default_rng(42).normal(0, 0.05, 100))
        sharpe = compute_sharpe(returns)
        assert isinstance(sharpe, float)

    def test_zero_std(self):
        assert compute_sharpe([0.0] * 20) == 0.0


class TestComputeMaxDrawdown:
    def test_empty(self):
        assert compute_max_drawdown([]) == 0.0

    def test_all_positive(self):
        dd = compute_max_drawdown([0.01, 0.02, 0.03])
        assert dd == 0.0

    def test_known_drawdown(self):
        returns = [0.1, -0.2, 0.1]
        dd = compute_max_drawdown(returns)
        assert dd > 0

    def test_nonnegative(self):
        import numpy as np
        returns = list(np.random.default_rng(1).normal(0, 0.02, 50))
        dd = compute_max_drawdown(returns)
        assert dd >= 0


class TestComputeWinRate:
    def test_empty(self):
        assert compute_win_rate([]) == 0.0

    def test_all_wins(self):
        trades = make_trades([0.01, 0.02, 0.03])
        assert compute_win_rate(trades) == 1.0

    def test_all_losses(self):
        trades = make_trades([-0.01, -0.02, -0.03])
        assert compute_win_rate(trades) == 0.0

    def test_half(self):
        trades = make_trades([0.01, -0.01, 0.01, -0.01])
        assert compute_win_rate(trades) == 0.5


class TestComputeTotalReturn:
    def test_empty(self):
        assert compute_total_return([]) == 0.0

    def test_compound(self):
        trades = make_trades([0.10, 0.10])
        total = compute_total_return(trades)
        assert abs(total - (1.1 * 1.1 - 1)) < 1e-9

    def test_loss(self):
        trades = make_trades([-0.05])
        assert abs(compute_total_return(trades) - (-0.05)) < 1e-9


class TestScore:
    def test_empty_trades(self):
        result = score([], GOAL)
        assert result["score"] == -1.0

    def test_output_keys(self):
        trades = make_trades([0.01] * 20)
        result = score(trades, GOAL)
        assert "score" in result
        assert "sharpe" in result
        assert "max_drawdown" in result
        assert "return" in result
        assert "win_rate" in result

    def test_score_range(self):
        trades = make_trades([0.01] * 20)
        result = score(trades, GOAL)
        assert -1.0 <= result["score"] <= 1.0

    def test_perfect_performance(self):
        trades = make_trades([0.03] * 30)
        result = score(trades, GOAL)
        assert result["score"] > 0

    def test_poor_performance(self):
        trades = make_trades([-0.02] * 30)
        result = score(trades, GOAL)
        assert result["score"] < 0

    def test_high_drawdown_penalized(self):
        good = make_trades([0.01] * 20)
        bad = make_trades([-0.05, 0.01] * 10)
        good_score = score(good, GOAL)["score"]
        bad_score = score(bad, GOAL)["score"]
        assert good_score > bad_score


class TestScoreByRegime:
    def test_splits_correctly(self):
        trades = (
            make_trades([0.01, 0.02], "TRENDING_UP") +
            make_trades([-0.01, -0.02], "RANGING")
        )
        result = score_by_regime(trades, GOAL)
        assert "TRENDING_UP" in result
        assert "RANGING" in result

    def test_each_has_score(self):
        trades = make_trades([0.01] * 10, "HIGH_VOLATILITY")
        result = score_by_regime(trades, GOAL)
        assert "HIGH_VOLATILITY" in result
        assert "score" in result["HIGH_VOLATILITY"]
