import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from utils import load_strategy, setup_logger
from indicators import compute_all

logger = setup_logger("strategy_engine")


def load_current_strategy() -> dict:
    return load_strategy()


def _rsi_signal(df: pd.DataFrame, threshold: float, direction: str) -> pd.Series:
    rsi = df["rsi"]
    if direction == "LONG":
        entry = rsi < threshold
        exit_ = rsi > (100 - threshold)
    else:
        entry = rsi > (100 - threshold)
        exit_ = rsi < threshold
    return entry, exit_


def _macd_signal(df: pd.DataFrame, direction: str):
    macd_hist = df["macd_hist"]
    prev_hist = macd_hist.shift(1)
    if direction == "LONG":
        entry = (prev_hist <= 0) & (macd_hist > 0)
        exit_ = (prev_hist >= 0) & (macd_hist < 0)
    else:
        entry = (prev_hist >= 0) & (macd_hist < 0)
        exit_ = (prev_hist <= 0) & (macd_hist > 0)
    return entry, exit_


def _ema_signal(df: pd.DataFrame, threshold: float, direction: str):
    fast = df["ema_20"]
    slow = df["ema_50"]
    if direction == "LONG":
        entry = fast > slow
        exit_ = fast < slow
    else:
        entry = fast < slow
        exit_ = fast > slow
    return entry, exit_


def generate_signals(df: pd.DataFrame, strategy: Optional[dict] = None) -> pd.DataFrame:
    if strategy is None:
        strategy = load_current_strategy()

    df = compute_all(df)
    indicator = strategy["entry"]["indicator"]
    threshold = float(strategy["entry"]["threshold"])
    direction = strategy.get("direction", "LONG")

    if indicator == "RSI":
        entry_sig, exit_sig = _rsi_signal(df, threshold, direction)
    elif indicator == "MACD":
        entry_sig, exit_sig = _macd_signal(df, direction)
    elif indicator == "EMA":
        entry_sig, exit_sig = _ema_signal(df, threshold, direction)
    else:
        logger.warning(f"Unknown indicator {indicator}, defaulting to RSI")
        entry_sig, exit_sig = _rsi_signal(df, threshold, direction)

    df["entry_signal"] = entry_sig.fillna(False)
    df["exit_signal"] = exit_sig.fillna(False)
    df["direction"] = direction
    df["strategy_version"] = strategy.get("version", "01")

    logger.debug(f"Signals: {entry_sig.sum()} entries, {exit_sig.sum()} exits over {len(df)} bars")
    return df
