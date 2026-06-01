import numpy as np
import pandas as pd
from typing import Tuple


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False, min_periods=period).mean()


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    return series - series.shift(period)


def rolling_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    log_returns = np.log(series / series.shift(1))
    return log_returns.rolling(window=period, min_periods=period).std() * np.sqrt(252)


def volume_change(volume: pd.Series, period: int = 1) -> pd.Series:
    return volume.pct_change(periods=period)


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["rsi"] = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    df["ema_20"] = ema(close, 20)
    df["ema_50"] = ema(close, 50)
    df["sma_20"] = sma(close, 20)
    df["sma_50"] = sma(close, 50)
    df["atr"] = atr(high, low, close, 14)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bollinger_bands(close, 20, 2.0)
    df["momentum"] = momentum(close, 10)
    df["volatility"] = rolling_volatility(close, 20)
    df["volume_change"] = volume_change(volume, 1)
    df["returns"] = close.pct_change()

    return df
