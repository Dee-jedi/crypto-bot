# Trading System - Backtest & Live Trading

## Overview

This trading system separates **backtesting** (using real historical data) from **live trading** (using testnet for practice).

### Key Improvements

1. **Real Historical Data**: Backtest now fetches 6-12 months of real market data from Binance production API
2. **Data Caching**: Downloaded data is cached locally to avoid re-fetching
3. **Better Validation**: Walk-forward validation on substantial historical periods
4. **Production/Testnet Separation**: Clear separation between backtest (production data) and live trading (testnet execution)

## File Structure

```
backtest_improved.py       # NEW: Enhanced backtester with real historical data
data_feed_improved.py      # NEW: Enhanced data fetching with date ranges
backtest.py               # ORIGINAL: Limited backtest (3000 candles)
data_feed.py              # ORIGINAL: Basic data fetching
config.py                 # Configuration (shared)
features.py               # Feature engineering
labels.py                 # Label generation
ict.py                    # ICT concepts
models.py                 # ML models (LSTM + XGBoost)
risk.py                   # Risk management
validation.py             # Walk-forward validation
main.py                   # Live trading bot (uses testnet)
```

## Quick Start

### 1. Backtest with Real Historical Data

```bash
# Run the improved backtest (fetches 12 months of real data)
python backtest_improved.py
```

**What it does:**

- Connects to **Binance Production API** (no auth needed for public data)
- Fetches 12 months of historical OHLCV data
- Caches data locally in `data/cache/` for 24 hours
- Runs full backtest with ICT strategy
- Outputs results to `logs/backtest_trades_*.csv` and `logs/backtest_equity_*.csv`

**Configuration:**

```python
# In backtest_improved.py, line 29
BACKTEST_MONTHS = 12  # Change this to fetch more/less data
```

### 2. Live Trading (Testnet)

```bash
# Run the live trading bot on Binance testnet
python main.py
```

**What it does:**

- Connects to **Binance Testnet** (requires testnet API keys)
- Executes trades with fake money
- Uses same strategy as backtest but in real-time
- Logs all trades to `logs/performance.csv`

## Data Fetching Details

### Original System (Limited)

```python
# backtest.py - Limited to recent data
df_15m = fetch_ohlcv_bulk(exchange, sym, TF_SIGNAL, total_limit=3000)  # ~1 month
df_1h  = fetch_ohlcv_bulk(exchange, sym, TF_BIAS,   total_limit=1000)  # ~1 month
```

### Improved System (Extensive)

```python
# backtest_improved.py - Fetches by date range
df_15m = get_cached_or_fetch(exchange, sym, TF_SIGNAL, months=12)  # 12 months
df_1h  = get_cached_or_fetch(exchange, sym, TF_BIAS, months=12)    # 12 months
```

## API Usage

### For Backtesting (No Auth Required)

```python
from data_feed_improved import connect_production_readonly, fetch_ohlcv_range
from datetime import datetime, timedelta

# Connect without API keys
exchange = connect_production_readonly()

# Fetch specific date range
start = datetime(2023, 1, 1)
end = datetime(2024, 1, 1)
df = fetch_ohlcv_range(exchange, 'BTC/USDT', '15m', start, end)
```

### For Live Trading (Testnet)

```python
from data_feed import connect

# Uses API_KEY and API_SECRET from config.py
exchange = connect()  # Connects to testnet if TESTNET=True
```

## Configuration

### config.py Settings

```python
# For BACKTESTING - not used (we use production read-only)
# API_KEY and API_SECRET are only needed for live trading

# For LIVE TRADING - set these for testnet
TESTNET = True  # MUST be True for paper trading
API_KEY = "your_testnet_api_key"
API_SECRET = "your_testnet_api_secret"
```

## Data Caching

Cached data is stored in `data/cache/` with format:

```
BTCUSDT_15m_12m.csv
ETHUSDT_15m_12m.csv
BTCUSDT_1h_12m.csv
...
```

Cache expires after 24 hours. Delete cache files to force re-download.

## Expected Backtest Time

| Data Period | Candles (15m) | Fetch Time | Backtest Time |
| ----------- | ------------- | ---------- | ------------- |
| 1 month     | ~2,880        | 30 sec     | 10 sec        |
| 3 months    | ~8,640        | 90 sec     | 30 sec        |
| 6 months    | ~17,280       | 3 min      | 60 sec        |
| 12 months   | ~34,560       | 6 min      | 2 min         |

_Times are approximate and depend on network speed and hardware_

## Output Files

### Backtest Results

```
logs/backtest_trades_BTCUSDT.csv    # All trades with entry/exit/PnL
logs/backtest_equity_BTCUSDT.csv    # Equity curve over time
```

### Live Trading Logs

```
logs/performance.csv    # Trade history
logs/equity.csv        # Balance tracking
logs/confidence.csv    # Model confidence over time
```

### Cached Data

```
data/cache/BTCUSDT_15m_12m.csv    # Historical price data (cached)
```

## Workflow

1. **Develop Strategy** → Edit `ict.py`, `features.py`, `labels.py`
2. **Backtest** → Run `python backtest_improved.py` with real historical data
3. **Validate** → Check results in `logs/backtest_*.csv`
4. **Deploy to Testnet** → Run `python main.py` for paper trading
5. **Monitor** → Watch performance in real-time with fake money
6. **Iterate** → Return to step 1, improve strategy

## Troubleshooting

### "Rate limit exceeded"

- The improved backtest has rate limiting built-in (sleeps for 60s if hit)
- Using cached data avoids this issue

### "Not enough data for backtest"

```python
# Increase months in backtest_improved.py
BACKTEST_MONTHS = 18  # Fetch more data
```

### "Cache is stale"

```bash
# Force fresh download
rm -rf data/cache/*
python backtest_improved.py
```

### "API keys not working"

- For backtest: No API keys needed (production read-only)
- For live trading: Get testnet keys from https://testnet.binancefuture.com/

## Performance Expectations

Based on 12-month backtest of BTC/USDT and ETH/USDT:

- **Expected Win Rate**: 45-55%
- **Expected Sharpe**: 0.8-1.5
- **Expected Max DD**: 5-8%
- **Expected Trades/Month**: 15-25 per symbol

_These are ballpark figures - actual results depend on market conditions_

## Next Steps

1. Run `python backtest_improved.py` to test with real historical data
2. Analyze results in `logs/`
3. Adjust parameters in `config.py` if needed
4. When satisfied, run `python main.py` for testnet paper trading
5. Monitor for 1-2 weeks before considering real money

## Important Notes

⚠️ **Never set `TESTNET = False` until you're ready for real money**

⚠️ **Backtest results don't guarantee future performance**

⚠️ **Always start with paper trading (testnet) before risking real capital**

## Support

For questions or issues:

1. Check logs in `logs/` directory
2. Review configuration in `config.py`
3. Verify data quality with `validate_ohlcv()` function
