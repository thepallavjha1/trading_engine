import pandas as pd
import numpy as np
from typing import Literal
from indicators import ema, rolling_volatility

RegimeType = Literal["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOLATILITY"]

VOLATILITY_THRESHOLD = 0.60
ADX_PERIOD = 14
TREND_EMA_FAST = 20
TREND_EMA_SLOW = 50


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr_s = tr.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(com=period - 1, adjust=False, min_periods=period).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(com=period - 1, adjust=False, min_periods=period).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    return adx, plus_di, minus_di


def detect_regime(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    vol = rolling_volatility(df["close"], 20)
    ema_fast = ema(df["close"], TREND_EMA_FAST)
    ema_slow = ema(df["close"], TREND_EMA_SLOW)
    adx, plus_di, minus_di = _compute_adx(df, ADX_PERIOD)

    regimes = []
    for i in range(len(df)):
        v = vol.iloc[i] if not pd.isna(vol.iloc[i]) else 0.0
        ef = ema_fast.iloc[i] if not pd.isna(ema_fast.iloc[i]) else np.nan
        es = ema_slow.iloc[i] if not pd.isna(ema_slow.iloc[i]) else np.nan
        adx_val = adx.iloc[i] if not pd.isna(adx.iloc[i]) else 0.0
        pdi = plus_di.iloc[i] if not pd.isna(plus_di.iloc[i]) else 0.0
        mdi = minus_di.iloc[i] if not pd.isna(minus_di.iloc[i]) else 0.0

        if v > VOLATILITY_THRESHOLD:
            regimes.append("HIGH_VOLATILITY")
        elif adx_val > 25:
            if pdi > mdi and not pd.isna(ef) and not pd.isna(es) and ef > es:
                regimes.append("TRENDING_UP")
            else:
                regimes.append("TRENDING_DOWN")
        else:
            regimes.append("RANGING")

    return pd.Series(regimes, index=df.index, name="regime")


def get_current_regime(df: pd.DataFrame) -> str:
    regimes = detect_regime(df)
    return regimes.iloc[-1] if not regimes.empty else "RANGING"
