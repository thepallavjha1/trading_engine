#!/usr/bin/env python3
import sys
import time
import signal
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils import setup_logger, load_goal, load_strategy
from market_data import get_ohlcv
from execution_engine import PaperTradeEngine
from trade_logger import log_trades, get_trade_count
from scoring_engine import score
from optimizer import run_optimization_cycle

logger = setup_logger("run")
_running = True


def handle_signal(sig, frame):
    global _running
    logger.info("Shutdown signal received")
    _running = False


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def run_cycle():
    goal = load_goal()
    strategy = load_strategy()
    asset = goal.get("asset", "BTC/USDT")
    timeframe = strategy.get("timeframe", "1h")
    lookback = int(strategy.get("lookback_candles", 200))
    reflection_every = int(goal.get("reflection_every", 5))

    logger.info(f"Fetching market data: {asset} [{timeframe}]")
    df = get_ohlcv(asset, timeframe, lookback, use_cache=True, max_cache_age_minutes=13)

    engine = PaperTradeEngine(strategy=strategy, goal=goal)
    trades = engine.run_backtest(df)

    if trades:
        log_trades(trades)
        perf = score(trades, goal)
        logger.info(
            f"Cycle complete: {len(trades)} trades | "
            f"return={perf['return']:.4f} sharpe={perf['sharpe']:.4f} "
            f"drawdown={perf['max_drawdown']:.4f} score={perf['score']:.4f}"
        )
        total_trades = get_trade_count()
        if total_trades % reflection_every == 0 and total_trades > 0:
            logger.info(f"Reflection threshold reached ({total_trades} trades). Running optimizer...")
            run_optimization_cycle()
    else:
        logger.info("No trades generated this cycle")


def main():
    parser = argparse.ArgumentParser(description="Local Self-Improving Trading Engine")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle and exit (used by GitHub Actions)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Local Self-Improving Trading Engine — STARTED")
    logger.info("=" * 60)

    goal = load_goal()
    interval_minutes = 15
    logger.info(f"Asset: {goal.get('asset')} | Mode: {'single-cycle' if args.once else 'continuous'}")

    run_cycle()

    if args.once:
        logger.info("Single-cycle mode complete.")
        return

    logger.info(f"Refresh interval: {interval_minutes}m | Press Ctrl+C to stop\n")
    while _running:
        sleep_secs = interval_minutes * 60
        logger.info(f"Sleeping {interval_minutes} minutes until next cycle...")
        for _ in range(sleep_secs):
            if not _running:
                break
            time.sleep(1)
        if _running:
            run_cycle()

    logger.info("Trading engine stopped cleanly.")


if __name__ == "__main__":
    main()
