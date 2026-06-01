import time
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from utils import setup_logger, MARKET_CACHE_DIR

logger = setup_logger("market_data")

BINANCE_BASE = "https://api.binance.com/api/v3"
TIMEFRAME_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


def _cache_path(symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "_")
    return MARKET_CACHE_DIR / f"{safe}_{timeframe}.csv"


def _load_cache(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    path = _cache_path(symbol, timeframe)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    except Exception:
        return None


def _save_cache(df: pd.DataFrame, symbol: str, timeframe: str) -> None:
    path = _cache_path(symbol, timeframe)
    df.to_csv(path)


def _binance_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


def fetch_binance_klines(
    symbol: str, timeframe: str = "1h", limit: int = 500
) -> Optional[pd.DataFrame]:
    bsym = _binance_symbol(symbol)
    url = f"{BINANCE_BASE}/klines"
    params = {"symbol": bsym, "interval": timeframe, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        logger.warning(f"Binance fetch failed for {symbol}: {e}")
        return None

    if not raw:
        return None

    cols = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    return df[["open", "high", "low", "close", "volume"]].copy()


def fetch_ccxt(symbol: str, timeframe: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv:
            return None
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        return df
    except Exception as e:
        logger.warning(f"CCXT fetch failed for {symbol}: {e}")
        return None


def fetch_yfinance(symbol: str, timeframe: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        tf_map = {"1h": "1h", "4h": "1h", "1d": "1d", "15m": "15m", "5m": "5m", "1m": "1m"}
        yf_tf = tf_map.get(timeframe, "1h")
        yf_sym = symbol.replace("/", "-")
        period_days = min(59, max(7, limit // 24 + 1))
        ticker = yf.Ticker(yf_sym)
        df = ticker.history(period=f"{period_days}d", interval=yf_tf)
        if df.empty:
            return None
        df = df.rename(columns=str.lower)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[["open", "high", "low", "close", "volume"]].copy()
        return df.tail(limit)
    except Exception as e:
        logger.warning(f"yfinance fetch failed for {symbol}: {e}")
        return None


def get_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 500,
    use_cache: bool = True,
    max_cache_age_minutes: int = 60,
) -> pd.DataFrame:
    if use_cache:
        cached = _load_cache(symbol, timeframe)
        if cached is not None and not cached.empty:
            last_ts = cached.index[-1]
            age = (datetime.now(tz=last_ts.tzinfo) - last_ts).total_seconds() / 60
            if age < max_cache_age_minutes:
                logger.info(f"Using cached data for {symbol} ({len(cached)} rows, {age:.1f}m old)")
                return cached

    df = fetch_ccxt(symbol, timeframe, limit)
    if df is None or df.empty:
        logger.info(f"CCXT unavailable, trying Binance REST for {symbol}")
        df = fetch_binance_klines(symbol, timeframe, limit)
    if df is None or df.empty:
        logger.info(f"Binance REST unavailable, trying yfinance for {symbol}")
        df = fetch_yfinance(symbol, timeframe, limit)
    if df is None or df.empty:
        raise RuntimeError(f"All data sources failed for {symbol}")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    _save_cache(df, symbol, timeframe)
    logger.info(f"Fetched {len(df)} candles for {symbol} [{timeframe}]")
    return df


def get_latest_price(symbol: str = "BTC/USDT") -> float:
    bsym = _binance_symbol(symbol)
    try:
        resp = requests.get(f"{BINANCE_BASE}/ticker/price", params={"symbol": bsym}, timeout=10)
        resp.raise_for_status()
        return float(resp.json()["price"])
    except Exception:
        pass
    try:
        df = get_ohlcv(symbol, "1m", 2, use_cache=False)
        return float(df["close"].iloc[-1])
    except Exception as e:
        raise RuntimeError(f"Cannot get latest price for {symbol}: {e}")
