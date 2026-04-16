import ccxt
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fetcher')

def fetch_history(symbol, timeframe, months):
    ex = ccxt.binance({'options': {'defaultType': 'future'}})
    # No API keys to bypass US regional api-key blockage
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=months * 30)
    
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    
    logger.info(f"Fetching {symbol} {timeframe} for {months} months...")
    all_bars = []
    current_ms = start_ms
    max_per_call = 1500
    
    while current_ms < end_ms:
        try:
            bars = ex.fetch_ohlcv(symbol, timeframe, since=current_ms, limit=max_per_call)
            if not bars:
                break
            all_bars.extend(bars)
            current_ms = bars[-1][0] + 1
            if len(all_bars) % 15000 == 0:
                logger.info(f"  Fetched {len(all_bars)} candles...")
        except Exception as e:
            logger.error(f"Error: {e}")
            break
            
    df = pd.DataFrame(all_bars, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df.set_index('time', inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    
    os.makedirs('data/cache', exist_ok=True)
    tag = symbol.replace('/', '')
    filepath = f"data/cache/{tag}_{timeframe}_{months}m.csv"
    df.to_csv(filepath)
    logger.info(f"Saved {filepath} ({len(df)} candles)")

if __name__ == '__main__':
    fetch_history('ETH/USDT', '1h', 60)
