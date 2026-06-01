import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from indicators import (
    sma, ema, rsi, macd, atr, bollinger_bands,
    momentum, rolling_volatility, volume_change, compute_all
)


def make_series(n=100, seed=42):
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.Series(prices, dtype=float)


def make_ohlcv(n=100, seed=42):
    close = make_series(n, seed)
    high = close + np.abs(np.random.default_rng(seed + 1).normal(0, 0.5, n))
    low = close - np.abs(np.random.default_rng(seed + 2).normal(0, 0.5, n))
    volume = np.random.default_rng(seed + 3).uniform(1000, 5000, n)
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestSMA:
    def test_length(self):
        s = make_series(50)
        result = sma(s, 10)
        assert len(result) == 50

    def test_first_valid(self):
        s = make_series(20)
        result = sma(s, 5)
        assert pd.isna(result.iloc[3])
        assert not pd.isna(result.iloc[4])

    def test_value(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, 3)
        assert abs(result.iloc[4] - 4.0) < 1e-9

    def test_all_same(self):
        s = pd.Series([5.0] * 20)
        result = sma(s, 5)
        assert all(result.dropna() == 5.0)


class TestEMA:
    def test_length(self):
        s = make_series(50)
        result = ema(s, 10)
        assert len(result) == 50

    def test_first_valid(self):
        s = make_series(20)
        result = ema(s, 5)
        assert pd.isna(result.iloc[3])
        assert not pd.isna(result.iloc[4])

    def test_reacts_to_change(self):
        s = pd.Series([100.0] * 20 + [200.0] * 20)
        result = ema(s, 5)
        assert result.iloc[-1] > result.iloc[19]


class TestRSI:
    def test_range(self):
        s = make_series(100)
        result = rsi(s, 14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_length(self):
        s = make_series(50)
        result = rsi(s, 14)
        assert len(result) == 50

    def test_flat_series(self):
        s = pd.Series([50.0] * 50)
        result = rsi(s, 14)
        assert len(result) == 50

    def test_all_up(self):
        s = pd.Series(np.arange(1.0, 51.0))
        result = rsi(s, 14)
        valid = result.dropna()
        assert (valid > 90).all()

    def test_all_down(self):
        s = pd.Series(np.arange(50.0, 0.0, -1.0))
        result = rsi(s, 14)
        valid = result.dropna()
        assert (valid < 10).all()


class TestMACD:
    def test_returns_three_series(self):
        s = make_series(100)
        ml, sig, hist = macd(s)
        assert len(ml) == len(sig) == len(hist) == 100

    def test_histogram_is_diff(self):
        s = make_series(100)
        ml, sig, hist = macd(s)
        diff = (ml - sig).dropna()
        hist_valid = hist.dropna()
        common = diff.index.intersection(hist_valid.index)
        np.testing.assert_allclose(diff[common].values, hist_valid[common].values, atol=1e-10)


class TestATR:
    def test_length(self):
        df = make_ohlcv(60)
        result = atr(df["high"], df["low"], df["close"], 14)
        assert len(result) == 60

    def test_nonnegative(self):
        df = make_ohlcv(60)
        result = atr(df["high"], df["low"], df["close"], 14)
        assert (result.dropna() >= 0).all()


class TestBollingerBands:
    def test_returns_three(self):
        s = make_series(60)
        upper, mid, lower = bollinger_bands(s, 20, 2.0)
        assert len(upper) == len(mid) == len(lower) == 60

    def test_ordering(self):
        s = make_series(60)
        upper, mid, lower = bollinger_bands(s, 20, 2.0)
        valid = ~(upper.isna() | mid.isna() | lower.isna())
        assert (upper[valid] >= mid[valid]).all()
        assert (mid[valid] >= lower[valid]).all()


class TestMomentum:
    def test_length(self):
        s = make_series(50)
        result = momentum(s, 10)
        assert len(result) == 50

    def test_value(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = momentum(s, 2)
        assert abs(result.iloc[2] - 2.0) < 1e-9


class TestRollingVolatility:
    def test_length(self):
        s = make_series(60)
        result = rolling_volatility(s, 20)
        assert len(result) == 60

    def test_nonnegative(self):
        s = make_series(60)
        result = rolling_volatility(s, 20)
        assert (result.dropna() >= 0).all()


class TestVolumeChange:
    def test_length(self):
        vol = pd.Series(np.random.uniform(1000, 5000, 50))
        result = volume_change(vol, 1)
        assert len(result) == 50


class TestComputeAll:
    def test_columns_present(self):
        df = make_ohlcv(200)
        result = compute_all(df)
        expected_cols = ["rsi", "macd", "macd_signal", "macd_hist",
                         "ema_20", "ema_50", "sma_20", "sma_50",
                         "atr", "bb_upper", "bb_mid", "bb_lower",
                         "momentum", "volatility", "volume_change", "returns"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_extra_rows(self):
        df = make_ohlcv(100)
        result = compute_all(df)
        assert len(result) == 100
