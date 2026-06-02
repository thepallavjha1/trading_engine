#!/usr/bin/env python3
import sys
import time
import signal
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils import setup_logger, load_goal, load_strategy
from market_data import get_ohlcv
from strategy_engine import generate_signals
from regime_detection import detect_regime
from trade_logger import log_trades, get_trade_count
from scoring_engine import score
from optimizer import run_optimization_cycle
from live_state import load_live_state, save_live_state

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
    timeframe = strategy.get("timeframe", "15m")
    lookback = int(strategy.get("lookback_candles", 200))
    reflection_every = int(goal.get("reflection_every", 5))
    initial_balance = float(goal.get("initial_balance", 10000))

    logger.info(f"Fetching market data: {asset} [{timeframe}]")
    df = get_ohlcv(asset, timeframe, lookback, use_cache=True, max_cache_age_minutes=13)

    df = generate_signals(df, strategy)
    regimes = detect_regime(df)
    df["regime"] = regimes

    latest = df.iloc[-1]
    latest_price = float(latest["close"])
    latest_ts = df.index[-1]

    stop_loss_pct = float(strategy.get("stop_loss_pct", 1.5)) / 100.0
    take_profit_pct = float(strategy.get("take_profit_pct", 4.0)) / 100.0
    direction = strategy.get("direction", "LONG")
    version = strategy.get("version", "02")

    state = load_live_state()
    balance = float(state.get("balance") or initial_balance)

    logger.info(
        f"Latest candle: {latest_ts} | price={latest_price:.2f} "
        f"rsi={latest.get('rsi', float('nan')):.1f} "
        f"entry_signal={bool(latest.get('entry_signal'))} "
        f"exit_signal={bool(latest.get('exit_signal'))} "
        f"in_position={state.get('in_position')}"
    )

    trades = []

    if state.get("in_position"):
        entry_price = float(state["entry_price"])
        qty = float(state["qty"])

        if direction == "LONG":
            hit_sl = latest_price <= entry_price * (1 - stop_loss_pct)
            hit_tp = latest_price >= entry_price * (1 + take_profit_pct)
        else:
            hit_sl = latest_price >= entry_price * (1 + stop_loss_pct)
            hit_tp = latest_price <= entry_price * (1 - take_profit_pct)

        should_exit = hit_sl or hit_tp or bool(latest.get("exit_signal", False))

        if should_exit:
            if hit_sl:
                exit_price = entry_price * (1 - stop_loss_pct) if direction == "LONG" else entry_price * (1 + stop_loss_pct)
                exit_reason = "stop_loss"
            elif hit_tp:
                exit_price = entry_price * (1 + take_profit_pct) if direction == "LONG" else entry_price * (1 - take_profit_pct)
                exit_reason = "take_profit"
            else:
                exit_price = latest_price
                exit_reason = "signal"

            if direction == "LONG":
                pnl = qty * (exit_price - entry_price)
                ret = (exit_price - entry_price) / entry_price
            else:
                pnl = qty * (entry_price - exit_price)
                ret = (entry_price - exit_price) / entry_price

            new_balance = balance + pnl

            trade = {
                "timestamp": latest_ts.isoformat(),
                "asset": asset,
                "entry_price": round(entry_price, 6),
                "exit_price": round(exit_price, 6),
                "return": round(ret, 6),
                "pnl": round(pnl, 4),
                "drawdown": 0.0,
                "hold_time": int(state.get("hold_bars", 0)),
                "regime": str(latest.get("regime", "RANGING")),
                "strategy_version": version,
                "direction": direction,
                "qty": round(qty, 8),
                "exit_reason": exit_reason,
            }
            trades.append(trade)
            save_live_state({"in_position": False, "balance": round(new_balance, 4)})
            logger.info(
                f"CLOSED {direction} @ {exit_price:.2f} | "
                f"reason={exit_reason} pnl={pnl:.2f} return={ret:.4f} balance={new_balance:.2f}"
            )
        else:
            hold_bars = int(state.get("hold_bars", 0)) + 1
            state["hold_bars"] = hold_bars
            save_live_state(state)
            current_return = (latest_price - entry_price) / entry_price if direction == "LONG" else (entry_price - latest_price) / entry_price
            logger.info(
                f"HOLDING {direction} entry={entry_price:.2f} "
                f"current={latest_price:.2f} unrealised={current_return:.4f} bars={hold_bars}"
            )
    else:
        if bool(latest.get("entry_signal", False)):
            pct = float(strategy.get("position_size_pct", 5)) / 100.0
            qty = (balance * pct) / latest_price
            new_state = {
                "in_position": True,
                "entry_price": latest_price,
                "entry_time": latest_ts.isoformat(),
                "qty": qty,
                "balance": round(balance, 4),
                "hold_bars": 0,
            }
            save_live_state(new_state)
            logger.info(f"ENTERED {direction} @ {latest_price:.2f} qty={qty:.8f} balance={balance:.2f}")
        else:
            logger.info(f"No entry signal this cycle — watching market")

    if trades:
        log_trades(trades)
        perf = score(trades, goal)
        logger.info(
            f"Trade logged: return={perf['return']:.4f} score={perf['score']:.4f}"
        )
        total_trades = get_trade_count()
        if total_trades % reflection_every == 0 and total_trades > 0:
            logger.info(f"Reflection threshold ({total_trades} trades) — running optimizer...")
            run_optimization_cycle()


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
