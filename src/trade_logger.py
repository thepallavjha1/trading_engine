import json
from pathlib import Path
from typing import List, Dict
from utils import append_jsonl, read_jsonl, TRADES_FILE, setup_logger

logger = setup_logger("trade_logger")


def log_trade(trade: dict) -> None:
    append_jsonl(TRADES_FILE, trade)
    logger.debug(f"Logged trade: {trade.get('asset')} version={trade.get('strategy_version')} return={trade.get('return'):.4f}")


def log_trades(trades: List[dict]) -> None:
    for t in trades:
        log_trade(t)
    logger.info(f"Logged {len(trades)} trades")


def load_all_trades() -> List[dict]:
    return read_jsonl(TRADES_FILE)


def load_trades_by_version(version: str) -> List[dict]:
    all_trades = load_all_trades()
    return [t for t in all_trades if t.get("strategy_version") == version]


def load_recent_trades(n: int = 20) -> List[dict]:
    all_trades = load_all_trades()
    return all_trades[-n:] if len(all_trades) >= n else all_trades


def get_trade_count() -> int:
    return len(load_all_trades())
