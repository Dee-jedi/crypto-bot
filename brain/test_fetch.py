import ccxt
import pandas as pd
from datetime import datetime, timezone, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fetch():
    exchange = ccxt.binance({
        'options': {'defaultType': 'future'},
        'enableRateLimit': True,
    })
    
    symbol = 'SOL/USDT'
    timeframe = '1h'
    # Last 1 day for testing
    start_date = datetime.now(timezone.utc) - timedelta(days=1)
    current_ms = int(start_date.timestamp() * 1000)
    
    try:
        logger.info(f"Testing fetch for {symbol}...")
        bars = exchange.fetch_ohlcv(symbol, timeframe, since=current_ms, limit=10)
        if bars:
            logger.info(f"Success! Fetched {len(bars)} bars.")
        else:
            logger.warning("Empty response.")
    except Exception as e:
        logger.error(f"Fetch failed: {e}")

if __name__ == '__main__':
    test_fetch()
