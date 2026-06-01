import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from utils import setup_logger, load_strategy, load_goal, now_iso
from regime_detection import detect_regime

logger = setup_logger("execution_engine")


class PaperTradeEngine:
    def __init__(self, strategy: Optional[dict] = None, goal: Optional[dict] = None):
        self.strategy = strategy or load_strategy()
        self.goal = goal or load_goal()
        self.balance = float(self.goal.get("initial_balance", 10000))
        self.initial_balance = self.balance
        self.equity_curve: List[Dict] = []
        self.open_position: Optional[Dict] = None
        self.trades: List[Dict] = []

    def reset(self):
        self.balance = self.initial_balance
        self.equity_curve = []
        self.open_position = None
        self.trades = []

    def _position_size(self, price: float) -> float:
        pct = float(self.strategy.get("position_size_pct", 5)) / 100.0
        notional = self.balance * pct
        return notional / price

    def run_backtest(self, df: pd.DataFrame) -> List[Dict]:
        from strategy_engine import generate_signals
        df = generate_signals(df, self.strategy)
        regimes = detect_regime(df)
        df["regime"] = regimes

        stop_loss_pct = float(self.strategy.get("stop_loss_pct", 2)) / 100.0
        take_profit_pct = float(self.strategy.get("take_profit_pct", 4)) / 100.0
        direction = self.strategy.get("direction", "LONG")
        version = self.strategy.get("version", "01")

        self.reset()
        trades = []
        equity = self.balance

        for i, (ts, row) in enumerate(df.iterrows()):
            price = float(row["close"])
            regime = str(row.get("regime", "RANGING"))

            if self.open_position is not None:
                pos = self.open_position
                entry_p = pos["entry_price"]
                qty = pos["qty"]
                hold_time = i - pos["bar_index"]

                if direction == "LONG":
                    current_return = (price - entry_p) / entry_p
                    hit_sl = price <= entry_p * (1 - stop_loss_pct)
                    hit_tp = price >= entry_p * (1 + take_profit_pct)
                else:
                    current_return = (entry_p - price) / entry_p
                    hit_sl = price >= entry_p * (1 + stop_loss_pct)
                    hit_tp = price <= entry_p * (1 - take_profit_pct)

                should_exit = hit_sl or hit_tp or bool(row.get("exit_signal", False))

                if should_exit:
                    if hit_sl:
                        exit_price = entry_p * (1 - stop_loss_pct) if direction == "LONG" else entry_p * (1 + stop_loss_pct)
                    elif hit_tp:
                        exit_price = entry_p * (1 + take_profit_pct) if direction == "LONG" else entry_p * (1 - take_profit_pct)
                    else:
                        exit_price = price

                    if direction == "LONG":
                        pnl = qty * (exit_price - entry_p)
                        ret = (exit_price - entry_p) / entry_p
                    else:
                        pnl = qty * (entry_p - exit_price)
                        ret = (entry_p - exit_price) / entry_p

                    self.balance += pnl
                    peak_balance = max(self.initial_balance, self.balance - pnl)
                    drawdown = max(0.0, (peak_balance - self.balance) / peak_balance) if peak_balance > 0 else 0.0

                    trade = {
                        "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                        "asset": self.goal.get("asset", "BTC/USDT"),
                        "entry_price": round(entry_p, 6),
                        "exit_price": round(exit_price, 6),
                        "return": round(ret, 6),
                        "pnl": round(pnl, 4),
                        "drawdown": round(drawdown, 6),
                        "hold_time": hold_time,
                        "regime": regime,
                        "strategy_version": version,
                        "direction": direction,
                        "qty": round(qty, 8),
                        "exit_reason": "stop_loss" if hit_sl else ("take_profit" if hit_tp else "signal"),
                    }
                    trades.append(trade)
                    self.trades.append(trade)
                    self.open_position = None

            if self.open_position is None and bool(row.get("entry_signal", False)):
                qty = self._position_size(price)
                self.open_position = {
                    "entry_price": price,
                    "qty": qty,
                    "bar_index": i,
                    "timestamp": ts,
                }

            equity = self.balance
            if self.open_position:
                ep = self.open_position["entry_price"]
                q = self.open_position["qty"]
                if direction == "LONG":
                    equity += q * (price - ep)
                else:
                    equity += q * (ep - price)

            self.equity_curve.append({
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "equity": round(equity, 4),
                "balance": round(self.balance, 4),
            })

        if self.open_position is not None:
            last_row = df.iloc[-1]
            price = float(last_row["close"])
            pos = self.open_position
            entry_p = pos["entry_price"]
            qty = pos["qty"]
            if direction == "LONG":
                pnl = qty * (price - entry_p)
                ret = (price - entry_p) / entry_p
            else:
                pnl = qty * (entry_p - price)
                ret = (entry_p - price) / entry_p
            self.balance += pnl
            last_ts = df.index[-1]
            trade = {
                "timestamp": last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts),
                "asset": self.goal.get("asset", "BTC/USDT"),
                "entry_price": round(entry_p, 6),
                "exit_price": round(price, 6),
                "return": round(ret, 6),
                "pnl": round(pnl, 4),
                "drawdown": 0.0,
                "hold_time": len(df) - 1 - pos["bar_index"],
                "regime": str(df.iloc[-1].get("regime", "RANGING")),
                "strategy_version": version,
                "direction": direction,
                "qty": round(qty, 8),
                "exit_reason": "end_of_data",
            }
            trades.append(trade)
            self.trades.append(trade)
            self.open_position = None

        logger.info(f"Backtest complete: {len(trades)} trades, final balance: {self.balance:.2f}")
        return trades
