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
    ├── run.py             # Live 15-min decide-and-execute loop
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

Runs continuously: every **15 minutes** it fetches the latest market data, evaluates
the most recently **closed** 15m candle, and makes a single live decision —
enter, hold, partial take-profit, or exit. State is persisted in
`data/live_state.json` so each cycle resumes exactly where the last one left off.
Optimization auto-triggers every `reflection_every` logged trades.

### Run exactly one decision cycle (used by GitHub Actions)
```bash
cd src
python run.py --once
```

Performs a single decide-and-execute cycle and exits. This is what the scheduled
GitHub Actions workflow calls every 15 minutes (cron `*/15 * * * *`).

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

## Decision Loop (every 15 minutes)

1. **Fetch** the latest OHLCV data for BTC/USDT (or configured asset)
2. **Generate signals** using RSI (configurable in `strategy.yaml`)
3. **Decide on the last closed 15m candle** — exactly one action per cycle:
   - If **flat** and an entry signal fires → **enter** at the current price
   - If **in a position** → check stop-loss, take-profit (TP1 partial / TP2),
     trailing fallback, or exit signal and act accordingly; otherwise **hold**
4. **Persist** the new position state to `data/live_state.json`
5. **Log** every closed trade with a regime tag to `data/trades.jsonl`
6. **Score** performance: 40% return + 30% drawdown + 30% Sharpe
7. Every `reflection_every` trades → **Reflection Engine** runs:
   - Detects primary weakness (return / drawdown / Sharpe)
   - Generates hypothesis
   - Changes **exactly one** parameter
   - Saves version snapshot
   - Logs reasoning to `reflections.jsonl`
8. **Repeat** 15 minutes later with the updated strategy and state

> The engine is fully **paper-trading**: no real orders are placed. "Execution"
> means updating the simulated balance and position in `data/live_state.json`.

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

`configs/strategy.yaml` (edit freely — changes take effect on the next 15-min cycle):
```yaml
version: "04"
entry:
  indicator: RSI        # RSI | MACD | EMA
  threshold: 45         # higher = enters on more setups (more aggressive)
direction: LONG
stop_loss_pct: 5
take_profit_pct: 10.5   # TP1 — exits 50% of the position
take_profit2_pct: 15    # TP2 — exits the remaining 50%
position_size_pct: 15   # % of balance risked per trade (more aggressive)
timeframe: "15m"        # candle size the engine decides on
lookback_candles: 200
```

## Running Tests

```bash
cd trading_engine
python -m pytest tests/ -v
```

## Deployment

The engine runs **serverless** on GitHub Actions and the dashboard is hosted on
**Streamlit Community Cloud**.

- **GitHub Actions** (`.github/workflows/trading_engine.yml`): a cron job runs
  `python run.py --once` every 15 minutes, generates a report, then commits and
  pushes any changes to `data/`, `configs/strategy.yaml`, `history/`, and
  `reports/`. Requires `contents: write` permission (already configured).
- **Streamlit** (`src/dashboard.py`): reads the committed data straight from the
  GitHub repo via `src/github_reader.py` when the `GITHUB_REPO` secret is set, so
  the dashboard always reflects the latest pushed cycle without needing its own
  copy of the data. Configure `GITHUB_REPO` in `.streamlit/secrets.toml` (local)
  or in the Streamlit Cloud app secrets.

## Supported Assets
- BTC/USDT
- ETH/USDT
- SOL/USDT

## Data Sources (priority order)
1. CCXT (Binance exchange via ccxt library)
2. Binance Public REST API
3. Yahoo Finance (yfinance)

All data is cached locally in `data/market_cache/`.
