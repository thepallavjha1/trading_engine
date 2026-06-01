#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from typing import List, Dict, Any, Optional
from utils import setup_logger, load_goal, load_strategy, now_iso
from trade_logger import load_recent_trades
from reflection_engine import run_reflection
from version_manager import apply_modification
from scoring_engine import score

logger = setup_logger("optimizer")


def run_optimization_cycle(n_recent: int = 50) -> Optional[Dict[str, Any]]:
    goal = load_goal()
    strategy = load_strategy()
    trades = load_recent_trades(n_recent)

    if len(trades) < 3:
        logger.info(f"Only {len(trades)} trades available. Need at least 3 to optimize.")
        return None

    hypothesis = run_reflection(trades, strategy, goal)
    if hypothesis is None:
        logger.warning("Reflection returned no hypothesis")
        return None

    var = hypothesis["changed_variable"]
    old_val = hypothesis["old_value"]
    new_val = hypothesis["new_value"]
    reason = hypothesis["reason"]
    effect = hypothesis["expected_effect"]

    updated_strategy = apply_modification(var, old_val, new_val, reason, effect)

    result = {
        "timestamp": now_iso(),
        "action": "optimization_cycle",
        "changed_variable": var,
        "old_value": old_val,
        "new_value": new_val,
        "reason": reason,
        "expected_effect": effect,
        "performance_before": hypothesis["performance"],
        "new_version": updated_strategy.get("version", "??"),
    }

    logger.info(
        f"Optimization complete: {var} {old_val} -> {new_val} "
        f"| new version: v{result['new_version']}"
    )
    return result


if __name__ == "__main__":
    result = run_optimization_cycle()
    if result:
        print(f"\nOptimization applied:")
        print(f"  Variable: {result['changed_variable']}")
        print(f"  {result['old_value']} -> {result['new_value']}")
        print(f"  Reason: {result['reason']}")
        print(f"  New version: v{result['new_version']}")
    else:
        print("No optimization performed (insufficient data or no change needed).")
