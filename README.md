# Local Self-Improving Trading Engine

A fully local paper-trading research platform that continuously improves its own strategy parameters using the scientific method.

## Architecture

```
trading_engine/
├── configs/
│   ├── goal.yaml          # Performance targets (return, drawdown, Sharpe)
│   └── strategy.yaml      # Active trading strategy (live config)
├── data/
│   ├── trades.jsonl       # Append-only trade log
│   ├── reflections.jsonl  # Optimization history
│   └── market_cache/      # Local OHLCV cache
├── history/
│   └── strategy_versions/ # Versioned strategy snapshots (v0001.yaml, ...)
├── reports/               # Markdown research reports
└── src/
    ├── run.py             # Main trading loop
    ├── optimizer.py       # One-shot optimization cycle
    ├── report_generator.py
    ├── dashboard.py       # Streamlit dashboard
    ├── market_data.py     # CCXT → Binance → yfinance
    ├── indicators.py      # RSI, MACD, EMA, SMA, ATR, BB, Momentum, Volatility
    ├── regime_detection.py
    ├── strategy_engine.py
    ├── execution_engine.py
    ├── trade_logger.py
    ├── scoring_engine.py
    ├── reflection_engine.py
    ├── version_manager.py
    └── utils.py
```

## Setup

```bash
cd trading_engine
pip install -r requirements.txt
```

## Running

### Start the paper trading engine
```bash
cd src
python run.py
```

Fetches market data, runs backtest over historical candles, logs trades, and auto-triggers optimization every `reflection_every` trades.

### Run one optimization cycle manually
```bash
cd src
python optimizer.py
```

Reads recent trades, detects weaknesses, changes exactly one strategy parameter.

### Generate a research report
```bash
cd src
python report_generator.py
```

Outputs a full markdown report to `reports/`.

### Launch the dashboard
```bash
cd src
streamlit run dashboard.py
```

Opens at `http://localhost:8501` with live charts.

## Self-Improvement Loop

1. **Fetch** OHLCV data for BTC/USDT (or configured asset)
2. **Generate signals** using RSI (configurable in `strategy.yaml`)
3. **Backtest** paper trades with stop-loss, take-profit, position sizing
4. **Log** every trade with regime tag
5. **Score** performance: 40% return + 30% drawdown + 30% Sharpe
6. Every `reflection_every` trades → **Reflection Engine** runs:
   - Detects primary weakness (return / drawdown / Sharpe)
   - Generates hypothesis
   - Changes **exactly one** parameter
   - Saves version snapshot
   - Logs reasoning to `reflections.jsonl`
7. **Repeat** with updated strategy

## Configuration

`configs/goal.yaml`:
```yaml
asset: "BTC/USDT"
target_return_30d: 0.05
max_drawdown: 0.08
min_sharpe: 1.2
reflection_every: 5
initial_balance: 10000
```

`configs/strategy.yaml` (edit freely — changes take effect on next run):
```yaml
version: "01"
entry:
  indicator: RSI
  threshold: 30
direction: LONG
stop_loss_pct: 2
take_profit_pct: 4
position_size_pct: 5
timeframe: "1h"
lookback_candles: 200
```

## Running Tests

```bash
cd trading_engine
python -m pytest tests/ -v
```

## Supported Assets
- BTC/USDT
- ETH/USDT
- SOL/USDT

## Data Sources (priority order)
1. CCXT (Binance exchange via ccxt library)
2. Binance Public REST API
3. Yahoo Finance (yfinance)

All data is cached locally in `data/market_cache/`.
