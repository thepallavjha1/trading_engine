import random
from typing import List, Dict, Any, Optional, Tuple
from utils import (
    setup_logger, load_goal, load_strategy,
    read_jsonl, REFLECTIONS_FILE, now_iso
)
from scoring_engine import score, score_by_regime

logger = setup_logger("reflection_engine")

VARIABLE_BOUNDS = {
    "rsi_threshold": (10, 45),
    "stop_loss_pct": (0.5, 8.0),
    "take_profit_pct": (1.0, 15.0),
    "position_size_pct": (1.0, 20.0),
}

STEP_SIZES = {
    "rsi_threshold": 2,
    "stop_loss_pct": 0.5,
    "take_profit_pct": 0.5,
    "position_size_pct": 1.0,
}


def _get_current_variable_values(strategy: dict) -> Dict[str, Any]:
    return {
        "rsi_threshold": float(strategy["entry"]["threshold"]),
        "stop_loss_pct": float(strategy.get("stop_loss_pct", 2)),
        "take_profit_pct": float(strategy.get("take_profit_pct", 4)),
        "position_size_pct": float(strategy.get("position_size_pct", 5)),
    }


def _detect_primary_weakness(perf: dict, goal: dict) -> str:
    target_return = float(goal.get("target_return_30d", 0.05))
    max_dd = float(goal.get("max_drawdown", 0.08))
    min_sharpe = float(goal.get("min_sharpe", 1.2))

    weaknesses = []
    if perf["return"] < target_return:
        gap = target_return - perf["return"]
        weaknesses.append(("low_return", gap))
    if perf["max_drawdown"] > max_dd:
        gap = perf["max_drawdown"] - max_dd
        weaknesses.append(("high_drawdown", gap))
    if perf["sharpe"] < min_sharpe:
        gap = min_sharpe - perf["sharpe"]
        weaknesses.append(("low_sharpe", gap))

    if not weaknesses:
        return "none"
    weaknesses.sort(key=lambda x: x[1], reverse=True)
    return weaknesses[0][0]


def _generate_hypothesis(weakness: str, current_values: dict, regime_scores: dict) -> Tuple[str, Any, Any, str, str]:
    if weakness == "low_return":
        dominant_regime = max(regime_scores, key=lambda r: -regime_scores[r].get("score", 0), default=None)
        var = random.choice(["take_profit_pct", "rsi_threshold"])
        current = current_values[var]
        lo, hi = VARIABLE_BOUNDS[var]
        step = STEP_SIZES[var]
        if var == "take_profit_pct":
            new_val = min(hi, current + step)
            reason = f"Return below target. Increasing take_profit_pct to capture larger winners."
            effect = "Higher take_profit_pct allows trades to run further, potentially increasing per-trade return."
        else:
            new_val = max(lo, current - step)
            reason = f"Return below target. Lowering RSI threshold to find stronger oversold entries."
            effect = "Lower RSI threshold means entries at deeper oversold conditions, potentially higher win rate."

    elif weakness == "high_drawdown":
        var = random.choice(["stop_loss_pct", "position_size_pct"])
        current = current_values[var]
        lo, hi = VARIABLE_BOUNDS[var]
        step = STEP_SIZES[var]
        new_val = max(lo, current - step)
        if var == "stop_loss_pct":
            reason = "Drawdown exceeds target. Tightening stop_loss_pct to limit per-trade loss."
            effect = "Tighter stop loss reduces maximum per-trade drawdown contribution."
        else:
            reason = "Drawdown exceeds target. Reducing position_size_pct to reduce portfolio impact."
            effect = "Smaller position size reduces total capital at risk per trade."

    elif weakness == "low_sharpe":
        var = random.choice(["stop_loss_pct", "take_profit_pct"])
        current = current_values[var]
        lo, hi = VARIABLE_BOUNDS[var]
        step = STEP_SIZES[var]
        if var == "stop_loss_pct":
            new_val = max(lo, current - step)
            reason = "Sharpe below target. Tightening stop loss to improve risk-adjusted return."
            effect = "Lower stop loss reduces losers, improving reward-to-risk ratio."
        else:
            new_val = min(hi, current + step)
            reason = "Sharpe below target. Increasing take profit to improve reward-to-risk ratio."
            effect = "Higher take profit increases average winner size relative to risk."

    else:
        var = random.choice(list(VARIABLE_BOUNDS.keys()))
        current = current_values[var]
        lo, hi = VARIABLE_BOUNDS[var]
        step = STEP_SIZES[var]
        direction = random.choice([-1, 1])
        new_val = current + direction * step
        new_val = max(lo, min(hi, new_val))
        reason = "Performance meets all targets. Exploring marginal improvements."
        effect = f"Exploratory adjustment to {var}."

    return var, current_values[var], new_val, reason, effect


def run_reflection(trades: List[dict], strategy: dict, goal: dict) -> Optional[Dict[str, Any]]:
    if not trades:
        logger.warning("No trades to reflect on")
        return None

    perf = score(trades, goal)
    regime_scores = score_by_regime(trades, goal)
    current_values = _get_current_variable_values(strategy)
    weakness = _detect_primary_weakness(perf, goal)

    var, old_val, new_val, reason, effect = _generate_hypothesis(weakness, current_values, regime_scores)

    if new_val == old_val:
        lo, hi = VARIABLE_BOUNDS[var]
        step = STEP_SIZES[var]
        new_val = min(hi, old_val + step) if old_val == lo else max(lo, old_val - step)

    hypothesis = {
        "timestamp": now_iso(),
        "weakness": weakness,
        "performance": perf,
        "regime_scores": {k: v["score"] for k, v in regime_scores.items()},
        "changed_variable": var,
        "old_value": old_val,
        "new_value": new_val,
        "reason": reason,
        "expected_effect": effect,
        "trade_count": len(trades),
    }

    logger.info(
        f"Reflection: weakness={weakness} | {var}: {old_val} -> {new_val} | {reason}"
    )
    return hypothesis


def load_reflections() -> List[dict]:
    return read_jsonl(REFLECTIONS_FILE)
