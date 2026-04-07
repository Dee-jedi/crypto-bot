import ccxt
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from config import API_KEY, API_SECRET, TESTNET

logger = logging.getLogger(__name__)


# ==================== CONNECTION ====================

def connect(use_testnet=None):
    """
    Connect to Binance.
    
    Args:
        use_testnet: Override config.TESTNET setting. 
                    None = use config, True = testnet, False = production
    """
    testnet_mode = TESTNET if use_testnet is None else use_testnet
    
    ex = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'options': {'defaultType': 'future'},
        'enableRateLimit': True,
    })
    
    if testnet_mode:
        ex.set_sandbox_mode(True)
        logger.info("Connected to Binance TESTNET")
    else:
        logger.info("Connected to Binance PRODUCTION")
    
    ex.load_markets()
    return ex


def connect_production_readonly():
    """
    Connect to production Binance for historical data only.
    No API keys required for public data.
    """
    ex = ccxt.binance({
        'options': {'defaultType': 'future'},
        'enableRateLimit': True,
    })
    ex.load_markets()
    logger.info("Connected to Binance production (read-only, no auth)")
    return ex


# ==================== OHLCV ====================

def fetch_ohlcv(exchange, symbol, tf, limit=500):
    """Basic OHLCV fetch for live trading."""
    bars = exchange.fetch_ohlcv(symbol, tf, limit=limit)
    df   = pd.DataFrame(bars, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df.set_index('time', inplace=True)
    return df


def fetch_ohlcv_bulk(exchange, symbol, tf, total_limit=3000):
    """
    Fetch more candles than the exchange per-call limit by paginating.
    Binance futures typically caps at 1500 per call.
    
    For backtesting, use fetch_ohlcv_range() instead for date-based fetching.
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

        if len(bars) < n:
            break

    df = pd.DataFrame(all_bars, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df.set_index('time', inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df


def fetch_ohlcv_range(exchange, symbol, timeframe, start_date, end_date=None, progress_callback=None):
    """
    Fetch OHLCV data for a specific date range.
    Ideal for backtesting with historical data.
    
    Args:
        exchange: ccxt exchange instance
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '15m', '1h')
        start_date: datetime object or string 'YYYY-MM-DD'
        end_date: datetime object or string 'YYYY-MM-DD' (default: now)
        progress_callback: Optional function(current, total) for progress updates
    
    Returns:
        DataFrame with OHLCV data
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if end_date is None:
        end_date = datetime.utcnow()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    
    logger.info(f"Fetching {symbol} {timeframe} from {start_date.date()} to {end_date.date()}")
    
    all_bars = []
    current_ms = start_ms
    max_per_call = 1500
    
    # Calculate expected number of candles
    timeframe_ms = {
        '1m': 60_000, '3m': 180_000, '5m': 300_000, '15m': 900_000,
        '30m': 1_800_000, '1h': 3_600_000, '2h': 7_200_000,
        '4h': 14_400_000, '1d': 86_400_000, '1w': 604_800_000
    }
    tf_ms = timeframe_ms.get(timeframe, 900_000)
    expected_candles = int((end_ms - start_ms) / tf_ms)
    
    logger.info(f"Expected ~{expected_candles:,} candles")
    
    batch_count = 0
    while current_ms < end_ms:
        try:
            bars = exchange.fetch_ohlcv(
                symbol, 
                timeframe,
                since=current_ms,
                limit=max_per_call
            )
            
            if not bars:
                break
            
            all_bars.extend(bars)
            batch_count += 1
            
            # Progress callback
            if progress_callback:
                progress_callback(len(all_bars), expected_candles)
            
            # Console progress indicator
            if batch_count % 10 == 0:
                progress = len(all_bars) / expected_candles * 100 if expected_candles > 0 else 0
                logger.info(f"  Fetched {len(all_bars):,} candles (~{progress:.1f}%)")
            
            # Move to next batch
            current_ms = bars[-1][0] + tf_ms
            
            # Safety check
            if len(bars) < max_per_call:
                break
                
        except ccxt.RateLimitExceeded:
            logger.warning("Rate limit hit, waiting 60 seconds...")
            import time
            time.sleep(60)
            continue
        except Exception as e:
            logger.error(f"Error fetching batch: {e}")
            if len(all_bars) > 0:
                logger.warning("Using partial data collected so far")
                break
            else:
                raise
    
    # Convert to DataFrame
    df = pd.DataFrame(all_bars, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df.set_index('time', inplace=True)
    
    # Remove duplicates and sort
    df = df[~df.index.duplicated(keep='last')].sort_index()
    
    # Filter to exact date range
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    logger.info(f"✓ Fetched {len(df):,} candles for {symbol} {timeframe}")
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


def fetch_funding_rate_history(exchange, symbol, start_date=None, limit=100):
    """
    Fetch historical funding rates.
    Useful for backtesting funding rate impact.
    """
    try:
        if start_date:
            since = int(start_date.timestamp() * 1000) if isinstance(start_date, datetime) else start_date
            history = exchange.fetch_funding_rate_history(symbol, since=since, limit=limit)
        else:
            history = exchange.fetch_funding_rate_history(symbol, limit=limit)
        
        df = pd.DataFrame(history)
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('time', inplace=True)
        return df[['fundingRate']]
    except Exception as e:
        logger.warning(f"Funding rate history fetch failed ({symbol}): {e}")
        return None


# ==================== OPEN INTEREST ====================

def fetch_open_interest(exchange, symbol):
    """Current open interest (single data point)."""
    try:
        oi = exchange.fetch_open_interest(symbol)
        return float(oi.get('openInterest', 0.0))
    except Exception as e:
        logger.warning(f"OI fetch failed ({symbol}): {e}")
        return 0.0


def fetch_open_interest_history(exchange, symbol, tf='15m', limit=200, start_date=None):
    """
    Returns OI as a DataFrame aligned to OHLCV timestamps.
    OI rising with price = genuine buyers.
    OI falling with price = short covering (weaker move).
    
    Note: Binance typically limits OI history to 90 days.
    """
    try:
        kwargs = {'limit': min(limit, 500)}
        if start_date:
            since = int(start_date.timestamp() * 1000) if isinstance(start_date, datetime) else start_date
            kwargs['since'] = since
            
        oi_data = exchange.fetch_open_interest_history(symbol, tf, **kwargs)
        
        if not oi_data:
            return None
            
        df = pd.DataFrame(oi_data)
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('time', inplace=True)
        return df[['openInterestAmount']].rename(columns={'openInterestAmount': 'OI'})
    except Exception as e:
        logger.warning(f"OI history fetch failed ({symbol}): {e}")
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


# ==================== DATA QUALITY ====================

def validate_ohlcv(df, symbol, timeframe):
    """
    Check for data quality issues.
    Returns (is_valid, issues_list)
    """
    issues = []
    
    # Check for missing data
    if df.empty:
        issues.append("Empty DataFrame")
        return False, issues
    
    # Check for nulls
    null_counts = df.isnull().sum()
    if null_counts.any():
        issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
    
    # Check for duplicates
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        issues.append(f"{duplicates} duplicate timestamps")
    
    # Check for gaps (missing candles)
    expected_freq = {
        '1m': '1T', '3m': '3T', '5m': '5T', '15m': '15T',
        '30m': '30T', '1h': '1H', '4h': '4H', '1d': '1D'
    }
    if timeframe in expected_freq:
        expected_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=expected_freq[timeframe])
        missing = len(expected_index) - len(df)
        if missing > 0:
            issues.append(f"~{missing} missing candles (gaps in data)")
    
    # Check for zero/negative values
    if (df[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
        issues.append("Zero or negative prices found")
    
    # Check OHLC logic
    invalid_ohlc = (
        (df['High'] < df['Low']) |
        (df['High'] < df['Open']) |
        (df['High'] < df['Close']) |
        (df['Low'] > df['Open']) |
        (df['Low'] > df['Close'])
    ).sum()
    if invalid_ohlc > 0:
        issues.append(f"{invalid_ohlc} candles with invalid OHLC relationships")
    
    is_valid = len(issues) == 0
    
    if not is_valid:
        logger.warning(f"Data quality issues for {symbol} {timeframe}: {', '.join(issues)}")
    else:
        logger.info(f"✓ Data quality check passed for {symbol} {timeframe}")
    
    return is_valid, issues


# ==================== UTILITIES ====================

def timeframe_to_minutes(tf):
    """Convert timeframe string to minutes."""
    mapping = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '1d': 1440, '1w': 10080
    }
    return mapping.get(tf, 15)


def estimate_candles(start_date, end_date, timeframe):
    """Estimate number of candles between two dates."""
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    minutes = (end_date - start_date).total_seconds() / 60
    tf_minutes = timeframe_to_minutes(timeframe)
    return int(minutes / tf_minutes)