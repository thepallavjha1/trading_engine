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

    # Use the last COMPLETED candle for signals (df.iloc[-2]); df.iloc[-1] is the
    # currently forming candle whose RSI may already have recovered even though the
    # prior closed candle triggered the condition — this was silently missing entries
    # on 15m data where RSI can swing 10+ points between candle close and next open.
    latest_price = float(df.iloc[-1]["close"])
    latest_ts = df.index[-1]
    signal_bar = df.iloc[-2]  # last closed candle — use this for all signal checks

    stop_loss_pct = float(strategy.get("stop_loss_pct", 5.0)) / 100.0
    take_profit_pct = float(strategy.get("take_profit_pct", 10.0)) / 100.0   # TP1 — 50% exit at 1:2 RR
    take_profit2_pct = float(strategy.get("take_profit2_pct", 15.0)) / 100.0 # TP2 — remaining 50% at 1:3 RR
    direction = strategy.get("direction", "LONG")
    version = strategy.get("version", "02")

    state = load_live_state()
    balance = float(state.get("balance") or initial_balance)
    tp1_hit = state.get("tp1_hit", False)

    logger.info(
        f"Latest candle: {latest_ts} | price={latest_price:.2f} "
        f"rsi={signal_bar.get('rsi', float('nan')):.1f} "
        f"entry_signal={bool(signal_bar.get('entry_signal'))} "
        f"exit_signal={bool(signal_bar.get('exit_signal'))} "
        f"in_position={state.get('in_position')} tp1_hit={tp1_hit}"
    )

    trades = []

    if state.get("in_position"):
        entry_price = float(state["entry_price"])
        qty = float(state["qty"])
        hold_bars = int(state.get("hold_bars", 0)) + 1

        # --- price levels ---
        if direction == "LONG":
            sl_price       = entry_price * (1 - stop_loss_pct)
            tp1_price      = entry_price * (1 + take_profit_pct)   # +10%
            tp2_price      = entry_price * (1 + take_profit2_pct)  # +15%
            fallback_price = entry_price * (1 + stop_loss_pct)     # +5% — trailing floor after TP1
            hit_sl       = latest_price <= sl_price
            hit_tp1      = latest_price >= tp1_price
            hit_tp2      = latest_price >= tp2_price
            hit_fallback = tp1_hit and latest_price <= fallback_price
        else:
            sl_price       = entry_price * (1 + stop_loss_pct)
            tp1_price      = entry_price * (1 - take_profit_pct)
            tp2_price      = entry_price * (1 - take_profit2_pct)
            fallback_price = entry_price * (1 - stop_loss_pct)
            hit_sl       = latest_price >= sl_price
            hit_tp1      = latest_price <= tp1_price
            hit_tp2      = latest_price <= tp2_price
            hit_fallback = tp1_hit and latest_price >= fallback_price

        signal_exit = bool(signal_bar.get("exit_signal", False))

        def _make_trade(exit_p, exit_r, exit_qty):
            if direction == "LONG":
                pnl = exit_qty * (exit_p - entry_price)
                ret = (exit_p - entry_price) / entry_price
            else:
                pnl = exit_qty * (entry_price - exit_p)
                ret = (entry_price - exit_p) / entry_price
            return {
                "timestamp": latest_ts.isoformat(),
                "asset": asset,
                "entry_price": round(entry_price, 6),
                "exit_price": round(exit_p, 6),
                "return": round(ret, 6),
                "pnl": round(pnl, 4),
                "drawdown": 0.0,
                "hold_time": hold_bars,
                "regime": str(signal_bar.get("regime", "RANGING")),
                "strategy_version": version,
                "direction": direction,
                "qty": round(exit_qty, 8),
                "exit_reason": exit_r,
            }, pnl

        if not tp1_hit and hit_tp1 and not hit_tp2:
            # ── PARTIAL EXIT: sell 50% at TP1 (1:2 RR) ──────────────────────
            partial_qty = qty / 2
            trade, pnl = _make_trade(tp1_price, "take_profit_1", partial_qty)
            trades.append(trade)
            new_balance = balance + pnl
            save_live_state({
                **state,
                "qty": round(qty - partial_qty, 8),
                "balance": round(new_balance, 4),
                "tp1_hit": True,
                "hold_bars": hold_bars,
            })
            logger.info(
                f"PARTIAL EXIT 50% {direction} @ {tp1_price:.2f} (TP1 1:2 RR) | "
                f"pnl={pnl:.2f} remaining={qty-partial_qty:.8f} balance={new_balance:.2f}"
            )

        elif hit_tp2 or hit_sl or hit_fallback or signal_exit:
            # ── FULL (or remaining) EXIT ──────────────────────────────────────
            if hit_tp2:
                exit_price_val = tp2_price
                exit_reason = "take_profit_2"
            elif hit_sl:
                exit_price_val = sl_price
                exit_reason = "stop_loss"
            elif hit_fallback:
                exit_price_val = fallback_price
                exit_reason = "fallback_1r"
            else:
                exit_price_val = latest_price
                exit_reason = "signal"

            trade, pnl = _make_trade(exit_price_val, exit_reason, qty)
            trades.append(trade)
            new_balance = balance + pnl
            save_live_state({"in_position": False, "balance": round(new_balance, 4)})
            logger.info(
                f"CLOSED {direction} @ {exit_price_val:.2f} | "
                f"reason={exit_reason} pnl={pnl:.2f} return={trade['return']:.4f} balance={new_balance:.2f}"
            )

        else:
            # ── HOLD ─────────────────────────────────────────────────────────
            state["hold_bars"] = hold_bars
            save_live_state(state)
            current_return = (
                (latest_price - entry_price) / entry_price if direction == "LONG"
                else (entry_price - latest_price) / entry_price
            )
            tp_status = f"TP1✓ → TP2@{tp2_price:.0f}/fallback@{fallback_price:.0f}" if tp1_hit else f"→ TP1@{tp1_price:.0f}"
            logger.info(
                f"HOLDING {direction} entry={entry_price:.2f} "
                f"current={latest_price:.2f} unrealised={current_return:.4f} "
                f"{tp_status} bars={hold_bars}"
            )

    else:
        if bool(signal_bar.get("entry_signal", False)):
            pct = float(strategy.get("position_size_pct", 5)) / 100.0
            qty = (balance * pct) / latest_price
            save_live_state({
                "in_position": True,
                "entry_price": latest_price,
                "entry_time": latest_ts.isoformat(),
                "qty": qty,
                "balance": round(balance, 4),
                "hold_bars": 0,
                "tp1_hit": False,
            })
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
