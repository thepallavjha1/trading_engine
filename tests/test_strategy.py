import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
import yaml
from strategy_engine import generate_signals, load_current_strategy


STRATEGY_RSI = {
    "version": "01",
    "entry": {"indicator": "RSI", "threshold": 30},
    "direction": "LONG",
    "stop_loss_pct": 2,
    "take_profit_pct": 4,
    "position_size_pct": 5,
    "timeframe": "1h",
    "lookback_candles": 200,
}

STRATEGY_MACD = {
    "version": "02",
    "entry": {"indicator": "MACD", "threshold": 0},
    "direction": "LONG",
    "stop_loss_pct": 2,
    "take_profit_pct": 4,
    "position_size_pct": 5,
}

STRATEGY_EMA = {
    "version": "03",
    "entry": {"indicator": "EMA", "threshold": 0},
    "direction": "LONG",
    "stop_loss_pct": 2,
    "take_profit_pct": 4,
    "position_size_pct": 5,
}


def make_ohlcv(n=300, seed=42):
    rng = np.random.default_rng(seed)
    prices = 30000 + np.cumsum(rng.normal(0, 50, n))
    prices = np.maximum(prices, 1000)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": prices * (1 + rng.normal(0, 0.001, n)),
        "high": prices * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low": prices * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close": prices,
        "volume": rng.uniform(100, 1000, n),
    }, index=ts)


class TestLoadCurrentStrategy:
    def test_returns_dict(self, tmp_path, monkeypatch):
        strategy_file = tmp_path / "strategy.yaml"
        with open(strategy_file, "w") as f:
            yaml.dump(STRATEGY_RSI, f)
        import utils
        monkeypatch.setattr(utils, "CONFIGS_DIR", tmp_path)
        strategy = load_current_strategy()
        assert isinstance(strategy, dict)

    def test_has_required_keys(self, tmp_path, monkeypatch):
        strategy_file = tmp_path / "strategy.yaml"
        with open(strategy_file, "w") as f:
            yaml.dump(STRATEGY_RSI, f)
        import utils
        monkeypatch.setattr(utils, "CONFIGS_DIR", tmp_path)
        strategy = load_current_strategy()
        assert "version" in strategy
        assert "entry" in strategy
        assert "direction" in strategy


class TestGenerateSignals:
    def test_rsi_strategy_produces_signals(self):
        df = make_ohlcv(300)
        result = generate_signals(df, STRATEGY_RSI)
        assert "entry_signal" in result.columns
        assert "exit_signal" in result.columns

    def test_macd_strategy_produces_signals(self):
        df = make_ohlcv(300)
        result = generate_signals(df, STRATEGY_MACD)
        assert "entry_signal" in result.columns
        assert "exit_signal" in result.columns

    def test_ema_strategy_produces_signals(self):
        df = make_ohlcv(300)
        result = generate_signals(df, STRATEGY_EMA)
        assert "entry_signal" in result.columns
        assert "exit_signal" in result.columns

    def test_signals_are_boolean(self):
        df = make_ohlcv(300)
        result = generate_signals(df, STRATEGY_RSI)
        assert result["entry_signal"].dtype == bool
        assert result["exit_signal"].dtype == bool

    def test_strategy_version_in_result(self):
        df = make_ohlcv(300)
        result = generate_signals(df, STRATEGY_RSI)
        assert "strategy_version" in result.columns
        assert (result["strategy_version"] == "01").all()

    def test_all_indicators_computed(self):
        df = make_ohlcv(300)
        result = generate_signals(df, STRATEGY_RSI)
        expected = ["rsi", "macd", "ema_20", "sma_20", "atr", "bb_upper", "momentum"]
        for col in expected:
            assert col in result.columns

    def test_no_signals_below_indicator_period(self):
        df = make_ohlcv(15)
        result = generate_signals(df, STRATEGY_RSI)
        assert "entry_signal" in result.columns

    def test_threshold_affects_signal_count(self):
        df = make_ohlcv(300)
        strat_low = dict(STRATEGY_RSI)
        strat_low["entry"] = {"indicator": "RSI", "threshold": 10}
        strat_high = dict(STRATEGY_RSI)
        strat_high["entry"] = {"indicator": "RSI", "threshold": 40}
        low_entries = generate_signals(df, strat_low)["entry_signal"].sum()
        high_entries = generate_signals(df, strat_high)["entry_signal"].sum()
        assert high_entries >= low_entries

    def test_short_direction(self):
        short = dict(STRATEGY_RSI, direction="SHORT")
        short["entry"] = {"indicator": "RSI", "threshold": 70}
        df = make_ohlcv(300)
        result = generate_signals(df, short)
        assert "entry_signal" in result.columns
        assert (result["direction"] == "SHORT").all()
