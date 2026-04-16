import ccxt
import pandas as pd
import numpy as np
import logging
from config import API_KEY, API_SECRET, TESTNET

logger = logging.getLogger(__name__)


# ==================== CONNECTION ====================

def connect():
    """
    Connects to the Binance Futures API. 
    Optimized for the modern Demo-FAPI endpoint (Dashboard balance trading).
    """
    params = {
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'options': {
            'defaultType': 'future',
            'fetchCurrencies': False
        },
    }
    
    ex = ccxt.binance(params)
    
    if TESTNET:
        logger.info("[CONNECTION] Activating Modern Demo-FAPI Hub...")
        # Point to the new Demo Mode API endpoint
        demo_url = 'https://demo-fapi.binance.com/fapi'
        ex.urls['api']['fapiPublic']  = demo_url
        ex.urls['api']['fapiPrivate'] = demo_url
        # Force the Demo Mode headers
        ex.headers = {
            'X-MBX-APIKEY': API_KEY
        }
            
    try:
        ex.load_markets()
        logger.info(f"[CONNECTION] Connected to {'Demo Mode' if TESTNET else 'Production'}")
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        raise
        
    return ex


# ==================== OHLCV ====================

def fetch_ohlcv(exchange, symbol, tf, limit=500):
    bars = exchange.fetch_ohlcv(symbol, tf, limit=limit)
    df   = pd.DataFrame(bars, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df.set_index('time', inplace=True)
    return df


def fetch_ohlcv_bulk(exchange, symbol, tf, total_limit=3000):
    """
    Fetch more candles than the exchange per-call limit by paginating.
    Binance futures typically caps at 1500 per call.
    """
    per_call = 1500
    all_bars = []
    end_time = None

    while len(all_bars) < total_limit:
        remaining = total_limit - len(all_bars)
        n = min(per_call, remaining)
        try:
            kwargs = {'limit': n}
            if end_time:
                kwargs['params'] = {'endTime': end_time}
            bars = exchange.fetch_ohlcv(symbol, tf, **kwargs)
        except Exception as e:
            logger.error(f"Bulk OHLCV fetch error: {e}")
            break

        if not bars:
            break

        all_bars = bars + all_bars
        end_time = bars[0][0] - 1   # Go further back

    df = pd.DataFrame(all_bars, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df.set_index('time', inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df


# ==================== FUNDING RATE ====================

def fetch_funding_rate(exchange, symbol):
    """
    Returns the current funding rate (float).
    Positive = longs pay shorts (crowded long / bearish pressure).
    Negative = shorts pay longs (crowded short / bullish pressure).
    """
    try:
        fr = exchange.fetch_funding_rate(symbol)
        return float(fr.get('fundingRate', 0.0))
    except Exception as e:
        logger.warning(f"Funding rate fetch failed ({symbol}): {e}")
        return 0.0


# ==================== OPEN INTEREST ====================

def fetch_open_interest(exchange, symbol, tf='15m', limit=200):
    """
    Returns OI as a Series aligned to OHLCV timestamps.
    OI rising with price = genuine buyers.
    OI falling with price = short covering (weaker move).
    """
    try:
        oi_data = exchange.fetch_open_interest_history(symbol, tf, limit=limit)
        df = pd.DataFrame(oi_data)
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('time', inplace=True)
        return df[['openInterestAmount']].rename(columns={'openInterestAmount': 'OI'})
    except Exception as e:
        logger.warning(f"OI fetch failed ({symbol}): {e}")
        return None


# ==================== CVD ====================

def compute_cvd(df):
    """
    Cumulative Volume Delta approximation from OHLCV.
    If Close >= Open: buy pressure = Volume.
    If Close <  Open: sell pressure = Volume.
    CVD = cumulative sum of (buy_vol - sell_vol).
    Divergence between CVD direction and price direction is a leading signal.
    """
    buy_vol  = np.where(df['Close'] >= df['Open'], df['Volume'], 0.0)
    sell_vol = np.where(df['Close'] <  df['Open'], df['Volume'], 0.0)
    delta    = pd.Series(buy_vol - sell_vol, index=df.index)
    return delta.cumsum()
