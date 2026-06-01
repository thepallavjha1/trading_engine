import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from utils import setup_logger, MARKET_CACHE_DIR

logger = setup_logger("market_data")

BINANCE_BASE = "https://api.binance.com/api/v3"
KRAKEN_BASE  = "https://api.kraken.com/0/public"
COINBASE_BASE = "https://api.exchange.coinbase.com/products"

_KRAKEN_SYMBOLS = {
    "BTC/USDT": "XBTUSD", "BTC/USD": "XBTUSD",
    "ETH/USDT": "ETHUSD", "ETH/USD": "ETHUSD",
    "SOL/USDT": "SOLUSD", "SOL/USD": "SOLUSD",
}
_KRAKEN_INTERVALS = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440,
}
_COINBASE_SYMBOLS = {
    "BTC/USDT": "BTC-USD", "BTC/USD": "BTC-USD",
    "ETH/USDT": "ETH-USD", "ETH/USD": "ETH-USD",
    "SOL/USDT": "SOL-USD", "SOL/USD": "SOL-USD",
}
_COINBASE_GRANULARITY = {
    "1m": 60, "5m": 300, "15m": 900,
    "1h": 3600, "4h": 21600, "1d": 86400,
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
    _cache_path(symbol, timeframe).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_cache_path(symbol, timeframe))


def _binance_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


def fetch_ccxt(symbol: str, timeframe: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv:
            return None
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("timestamp")
    except Exception as e:
        logger.warning(f"CCXT fetch failed for {symbol}: {e}")
        return None


def fetch_binance_klines(symbol: str, timeframe: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
    try:
        bsym = _binance_symbol(symbol)
        resp = requests.get(
            f"{BINANCE_BASE}/klines",
            params={"symbol": bsym, "interval": timeframe, "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            return None
        cols = ["timestamp","open","high","low","close","volume",
                "close_time","quote_vol","trades","taker_base","taker_quote","ignore"]
        df = pd.DataFrame(raw, columns=cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c])
        return df[["open","high","low","close","volume"]]
    except Exception as e:
        logger.warning(f"Binance fetch failed for {symbol}: {e}")
        return None


def fetch_kraken(symbol: str, timeframe: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
    pair = _KRAKEN_SYMBOLS.get(symbol)
    if not pair:
        logger.warning(f"Kraken: no mapping for {symbol}")
        return None
    interval = _KRAKEN_INTERVALS.get(timeframe, 60)
    since = int(time.time()) - (limit + 50) * interval * 60
    try:
        resp = requests.get(
            f"{KRAKEN_BASE}/OHLC",
            params={"pair": pair, "interval": interval, "since": since},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            logger.warning(f"Kraken API error: {data['error']}")
            return None
        result = data.get("result", {})
        candles = result.get(pair) or result.get(next(iter(result), None))
        if not candles:
            return None
        df = pd.DataFrame(candles, columns=["time","open","high","low","close","vwap","volume","count"])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c])
        return df[["open","high","low","close","volume"]].tail(limit)
    except Exception as e:
        logger.warning(f"Kraken fetch failed for {symbol}: {e}")
        return None


def fetch_coinbase(symbol: str, timeframe: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
    product = _COINBASE_SYMBOLS.get(symbol)
    if not product:
        logger.warning(f"Coinbase: no mapping for {symbol}")
        return None
    granularity = _COINBASE_GRANULARITY.get(timeframe, 3600)
    end = datetime.utcnow()
    start = end - timedelta(seconds=granularity * (limit + 10))
    try:
        resp = requests.get(
            f"{COINBASE_BASE}/{product}/candles",
            params={
                "granularity": granularity,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            headers={"Accept": "application/json"},
            timeout=20,
        )
        resp.raise_for_status()
        raw = resp.json()
        if not raw or isinstance(raw, dict):
            return None
        df = pd.DataFrame(raw, columns=["time","low","high","open","close","volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.sort_values("time").set_index("time")
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c])
        return df[["open","high","low","close","volume"]].tail(limit)
    except Exception as e:
        logger.warning(f"Coinbase fetch failed for {symbol}: {e}")
        return None


def fetch_yfinance(symbol: str, timeframe: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        # Yahoo Finance uses BTC-USD, not BTC-USDT
        yf_sym = symbol.replace("/USDT", "-USD").replace("/USD", "-USD").replace("/", "-")
        tf_map = {"1m":"1m","5m":"5m","15m":"15m","1h":"1h","4h":"1h","1d":"1d"}
        yf_tf = tf_map.get(timeframe, "1h")
        period_days = min(59, max(7, limit // 24 + 1))
        ticker = yf.Ticker(yf_sym)
        df = ticker.history(period=f"{period_days}d", interval=yf_tf)
        if df.empty:
            return None
        df = df.rename(columns=str.lower)
        df.index = pd.to_datetime(df.index, utc=True)
        return df[["open","high","low","close","volume"]].tail(limit)
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

    fetchers = [
        ("CCXT",     lambda: fetch_ccxt(symbol, timeframe, limit)),
        ("Binance",  lambda: fetch_binance_klines(symbol, timeframe, limit)),
        ("Kraken",   lambda: fetch_kraken(symbol, timeframe, limit)),
        ("Coinbase", lambda: fetch_coinbase(symbol, timeframe, limit)),
        ("yfinance", lambda: fetch_yfinance(symbol, timeframe, limit)),
    ]

    df = None
    for name, fetcher in fetchers:
        logger.info(f"Trying {name} for {symbol}...")
        df = fetcher()
        if df is not None and not df.empty:
            logger.info(f"{name} succeeded: {len(df)} candles for {symbol} [{timeframe}]")
            break
        logger.info(f"{name} unavailable, trying next source")

    if df is None or df.empty:
        raise RuntimeError(f"All data sources failed for {symbol}")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    _save_cache(df, symbol, timeframe)
    return df


def get_latest_price(symbol: str = "BTC/USDT") -> float:
    try:
        bsym = _binance_symbol(symbol)
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
