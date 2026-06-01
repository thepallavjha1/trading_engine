import numpy as np
import pandas as pd
from typing import List, Dict, Any
from utils import setup_logger, safe_divide

logger = setup_logger("scoring_engine")

WEIGHT_RETURN = 0.40
WEIGHT_DRAWDOWN = 0.30
WEIGHT_SHARPE = 0.30


def compute_sharpe(returns: List[float], risk_free_rate: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    r = np.array(returns)
    excess = r - risk_free_rate / 252
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))


def compute_max_drawdown(returns: List[float]) -> float:
    if not returns:
        return 0.0
    cum = np.cumprod(1 + np.array(returns))
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(abs(dd.min()))


def compute_win_rate(trades: List[dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("return", 0) > 0)
    return wins / len(trades)


def compute_total_return(trades: List[dict]) -> float:
    if not trades:
        return 0.0
    returns = [t.get("return", 0.0) for t in trades]
    cum = 1.0
    for r in returns:
        cum *= (1 + r)
    return cum - 1.0


def score(trades: List[dict], goal: dict) -> Dict[str, float]:
    if not trades:
        return {
            "score": -1.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "return": 0.0,
            "win_rate": 0.0,
        }

    returns = [t.get("return", 0.0) for t in trades]
    total_return = compute_total_return(trades)
    sharpe = compute_sharpe(returns)
    max_dd = compute_max_drawdown(returns)
    win_rate = compute_win_rate(trades)

    target_return = float(goal.get("target_return_30d", 0.05))
    max_allowed_dd = float(goal.get("max_drawdown", 0.08))
    min_sharpe = float(goal.get("min_sharpe", 1.2))

    return_score = min(1.0, total_return / target_return) if target_return > 0 else 0.0
    return_score = max(-1.0, return_score)

    dd_score = 1.0 - min(1.0, max_dd / max_allowed_dd) if max_allowed_dd > 0 else 0.0
    if max_dd > max_allowed_dd:
        dd_score = -1.0

    sharpe_score = min(1.0, sharpe / min_sharpe) if min_sharpe > 0 else 0.0
    sharpe_score = max(-1.0, sharpe_score)

    composite = (
        WEIGHT_RETURN * return_score
        + WEIGHT_DRAWDOWN * dd_score
        + WEIGHT_SHARPE * sharpe_score
    )
    composite = float(np.clip(composite, -1.0, 1.0))

    result = {
        "score": round(composite, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "return": round(total_return, 4),
        "win_rate": round(win_rate, 4),
        "trade_count": len(trades),
        "return_score": round(return_score, 4),
        "dd_score": round(dd_score, 4),
        "sharpe_score": round(sharpe_score, 4),
    }
    logger.info(f"Score: {composite:.4f} | return={total_return:.4f} sharpe={sharpe:.4f} dd={max_dd:.4f}")
    return result


def score_by_regime(trades: List[dict], goal: dict) -> Dict[str, Dict]:
    regimes = {}
    for t in trades:
        r = t.get("regime", "UNKNOWN")
        regimes.setdefault(r, []).append(t)
    return {regime: score(rtrades, goal) for regime, rtrades in regimes.items()}
