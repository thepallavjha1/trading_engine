import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from execution_engine import PaperTradeEngine


STRATEGY = {
    "version": "01",
    "entry": {"indicator": "RSI", "threshold": 30},
    "direction": "LONG",
    "stop_loss_pct": 2,
    "take_profit_pct": 4,
    "position_size_pct": 5,
    "timeframe": "1h",
    "lookback_candles": 200,
}

GOAL = {
    "asset": "BTC/USDT",
    "initial_balance": 10000,
    "target_return_30d": 0.05,
    "max_drawdown": 0.08,
    "min_sharpe": 1.2,
    "reflection_every": 5,
}


def make_ohlcv(n=300, seed=42, trend="flat"):
    rng = np.random.default_rng(seed)
    if trend == "up":
        prices = 30000 + np.cumsum(np.abs(rng.normal(50, 20, n)))
    elif trend == "down":
        prices = 30000 + np.cumsum(-np.abs(rng.normal(50, 20, n)))
    else:
        prices = 30000 + np.cumsum(rng.normal(0, 50, n))

    prices = np.maximum(prices, 1000)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": prices * (1 + rng.normal(0, 0.001, n)),
        "high": prices * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low": prices * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close": prices,
        "volume": rng.uniform(100, 1000, n),
    }, index=ts)
    return df


class TestPaperTradeEngine:
    def test_initializes_balance(self):
        engine = PaperTradeEngine(strategy=STRATEGY, goal=GOAL)
        assert engine.balance == 10000.0

    def test_reset_restores_balance(self):
        engine = PaperTradeEngine(strategy=STRATEGY, goal=GOAL)
        engine.balance = 9000
        engine.reset()
        assert engine.balance == 10000.0

    def test_backtest_returns_list(self):
        engine = PaperTradeEngine(strategy=STRATEGY, goal=GOAL)
        df = make_ohlcv(300)
        trades = engine.run_backtest(df)
        assert isinstance(trades, list)

    def test_trade_has_required_fields(self):
        engine = PaperTradeEngine(strategy=STRATEGY, goal=GOAL)
        df = make_ohlcv(300)
        trades = engine.run_backtest(df)
        if trades:
            t = trades[0]
            assert "timestamp" in t
            assert "asset" in t
            assert "entry_price" in t
            assert "exit_price" in t
            assert "return" in t
            assert "pnl" in t
            assert "drawdown" in t
            assert "hold_time" in t
            assert "regime" in t
            assert "strategy_version" in t

    def test_equity_curve_built(self):
        engine = PaperTradeEngine(strategy=STRATEGY, goal=GOAL)
        df = make_ohlcv(300)
        engine.run_backtest(df)
        assert len(engine.equity_curve) == len(df)

    def test_no_open_position_after_backtest(self):
        engine = PaperTradeEngine(strategy=STRATEGY, goal=GOAL)
        df = make_ohlcv(300)
        engine.run_backtest(df)
        assert engine.open_position is None

    def test_strategy_version_in_trades(self):
        engine = PaperTradeEngine(strategy=STRATEGY, goal=GOAL)
        df = make_ohlcv(300)
        trades = engine.run_backtest(df)
        for t in trades:
            assert t["strategy_version"] == "01"

    def test_stop_loss_respected(self):
        engine = PaperTradeEngine(strategy=STRATEGY, goal=GOAL)
        df = make_ohlcv(300)
        trades = engine.run_backtest(df)
        sl_pct = STRATEGY["stop_loss_pct"] / 100
        for t in trades:
            if t.get("exit_reason") == "stop_loss":
                ratio = abs((t["exit_price"] - t["entry_price"]) / t["entry_price"])
                assert ratio <= sl_pct + 0.001

    def test_take_profit_respected(self):
        engine = PaperTradeEngine(strategy=STRATEGY, goal=GOAL)
        df = make_ohlcv(300)
        trades = engine.run_backtest(df)
        tp_pct = STRATEGY["take_profit_pct"] / 100
        for t in trades:
            if t.get("exit_reason") == "take_profit":
                ratio = abs((t["exit_price"] - t["entry_price"]) / t["entry_price"])
                assert ratio <= tp_pct + 0.001

    def test_position_size_scales_with_balance(self):
        engine = PaperTradeEngine(strategy=STRATEGY, goal=GOAL)
        price = 30000.0
        qty = engine._position_size(price)
        expected_notional = 10000 * 0.05
        expected_qty = expected_notional / price
        assert abs(qty - expected_qty) < 1e-9

    def test_regime_tagged_in_trades(self):
        engine = PaperTradeEngine(strategy=STRATEGY, goal=GOAL)
        df = make_ohlcv(300)
        trades = engine.run_backtest(df)
        valid_regimes = {"TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOLATILITY"}
        for t in trades:
            assert t.get("regime") in valid_regimes

    def test_drawdown_nonnegative(self):
        engine = PaperTradeEngine(strategy=STRATEGY, goal=GOAL)
        df = make_ohlcv(300)
        trades = engine.run_backtest(df)
        for t in trades:
            assert t["drawdown"] >= 0

    def test_hold_time_nonnegative(self):
        engine = PaperTradeEngine(strategy=STRATEGY, goal=GOAL)
        df = make_ohlcv(300)
        trades = engine.run_backtest(df)
        for t in trades:
            assert t["hold_time"] >= 0

    def test_short_direction(self):
        short_strategy = dict(STRATEGY, direction="SHORT")
        short_strategy["entry"] = {"indicator": "RSI", "threshold": 70}
        engine = PaperTradeEngine(strategy=short_strategy, goal=GOAL)
        df = make_ohlcv(300)
        trades = engine.run_backtest(df)
        for t in trades:
            assert t["direction"] == "SHORT"
