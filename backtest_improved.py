"""
Standalone backtester with real historical data from Binance production API.

Usage:
    python backtest_improved.py

Key improvements:
    - Uses production Binance API for real historical data
    - Configurable date ranges (default: 12 months)
    - Fetches sufficient data for robust backtesting
    - Automatic data caching to avoid re-fetching
    - Progress indicators for long-running fetches

Produces:
    - Console summary
    - logs/backtest_trades_{SYMBOL}.csv
    - logs/backtest_equity_{SYMBOL}.csv
    - data/cache/{SYMBOL}_{TF}_historical.csv (cached data)
"""

import os
import csv
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import ccxt

from config import (
    SYMBOLS, TF_SIGNAL, TF_BIAS,
    RISK_PERCENT, FEE_MAKER, FEE_TAKER,
    MIN_RR, LOG_DIR,
)
from features import build_features, htf_bias, FEAT_COLS
from labels import build_labels, optimize_multipliers
from ict import break_of_structure, liquidity_sweep, fvg, order_block, confluence_score
from validation import trade_stats, sharpe_ratio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SLIPPAGE = 0.0005   # 0.05% per side
LOOKAHEAD = 12

# Historical data settings
BACKTEST_MONTHS = 12  # Fetch 12 months of historical data
CACHE_DIR = 'data/cache'


# ==================== DATA FETCHING ====================

def connect_production():
    """
    Connect to PRODUCTION Binance for historical data.
    No API keys needed for public historical data.
    """
    ex = ccxt.binance({
        'options': {'defaultType': 'future'},
        'enableRateLimit': True,  # Important for bulk fetching
    })
    ex.load_markets()
    logger.info("Connected to Binance production API for historical data")
    return ex


def fetch_historical_range(exchange, symbol, timeframe, start_date, end_date=None):
    """
    Fetch OHLCV data for a specific date range.
    
    Args:
        exchange: ccxt exchange instance
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '15m', '1h')
        start_date: datetime object or string 'YYYY-MM-DD'
        end_date: datetime object or string 'YYYY-MM-DD' (default: now)
    
    Returns:
        DataFrame with OHLCV data
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    elif getattr(start_date, 'tzinfo', None) is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
        
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    elif getattr(end_date, 'tzinfo', None) is None:
        end_date = end_date.replace(tzinfo=timezone.utc)
    
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
        '4h': 14_400_000, '1d': 86_400_000
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
            
            # Progress indicator
            if batch_count % 10 == 0:
                progress = len(all_bars) / expected_candles * 100 if expected_candles > 0 else 0
                logger.info(f"  Fetched {len(all_bars):,} candles (~{progress:.1f}%)")
            
            # Move to next batch
            current_ms = bars[-1][0] + 1  # Standard CCXT pagination using +1ms
                
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


def get_cached_or_fetch(exchange, symbol, timeframe, months=12):
    """
    Check if cached historical data exists and is recent.
    If not, fetch fresh data and cache it.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    cache_file = os.path.join(
        CACHE_DIR, 
        f"{symbol.replace('/', '')}_{timeframe}_{months}m.csv"
    )
    
    # Check cache
    if os.path.exists(cache_file):
        cache_age_hours = (datetime.now().timestamp() - os.path.getmtime(cache_file)) / 3600
        if cache_age_hours < 24:  # Use cache if less than 24 hours old
            logger.info(f"Loading cached data: {cache_file} (age: {cache_age_hours:.1f}h)")
            df = pd.read_csv(cache_file, index_col='time', parse_dates=True)
            return df
        else:
            logger.info(f"Cache expired (age: {cache_age_hours:.1f}h), fetching fresh data")
    
    # Fetch fresh data
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=months * 30)
    
    df = fetch_historical_range(exchange, symbol, timeframe, start_date, end_date)
    
    # Cache it
    df.to_csv(cache_file)
    logger.info(f"Cached data saved: {cache_file}")
    
    return df


def fetch_open_interest_historical(exchange, symbol, tf='15m', months=12):
    """
    Fetch historical Open Interest data.
    Note: OI history may not go back as far as OHLCV.
    """
    try:
        # OI data typically limited to recent history (90 days on Binance)
        days = min(months * 30, 90)  # Cap at 90 days
        limit = int(days * 24 * (60 / int(tf.replace('m', '').replace('h', ''))))
        
        logger.info(f"Fetching {days} days of OI data for {symbol}")
        oi_data = exchange.fetch_open_interest_history(symbol, tf, limit=min(limit, 500))
        
        if not oi_data:
            return None
            
        df = pd.DataFrame(oi_data)
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('time', inplace=True)
        return df[['openInterestAmount']].rename(columns={'openInterestAmount': 'OI'})
    except Exception as e:
        logger.warning(f"OI fetch failed ({symbol}): {e}")
        return None


# ==================== BACKTESTER ====================

class Backtester:
    def __init__(self, symbol, df_15m, df_1h, tp_mult, sl_mult):
        self.symbol   = symbol
        self.df       = df_15m
        self.df_1h    = df_1h
        self.tp_mult  = tp_mult
        self.sl_mult  = sl_mult
        self.trades   = []
        self.equity   = [1.0]
        self.balance  = 1.0   # Normalised to 1.0

    def run(self):
        df     = self.df
        df_1h  = self.df_1h
        closes = df['Close'].values
        highs  = df['High'].values
        lows   = df['Low'].values
        atrs   = df['ATR'].values
        n      = len(df)
        open_trade = None

        logger.info(f"Running backtest on {n:,} candles...")

        for i in range(200, n - LOOKAHEAD):   # 200 warmup bars
            price = closes[i]
            atr   = atrs[i]
            ts    = df.index[i]

            # ---- Manage open trade ----
            if open_trade:
                h = highs[i]
                l = lows[i]
                t = open_trade

                # Partial close at 1R
                if not t.get('partial') and (
                    (t['type'] == 'LONG'  and h >= t['entry'] + atr) or
                    (t['type'] == 'SHORT' and l <= t['entry'] - atr)
                ):
                    t['partial'] = True
                    t['sl']      = t['entry']   # Move to breakeven
                    t['size']   *= 0.5

                # Trail stop (LONG)
                if t['type'] == 'LONG' and h >= t['entry'] + 2 * atr:
                    t['sl'] = max(t['sl'], h - atr)

                # Trail stop (SHORT)
                if t['type'] == 'SHORT' and l <= t['entry'] - 2 * atr:
                    t['sl'] = min(t['sl'], l + atr)

                # Exit: SL
                if (t['type'] == 'LONG'  and l <= t['sl']) or \
                   (t['type'] == 'SHORT' and h >= t['sl']):
                    exit_p = t['sl'] * (1 - SLIPPAGE)
                    self._record(t, exit_p, 'SL', ts)
                    open_trade = None
                    continue

                # Exit: TP
                if (t['type'] == 'LONG'  and h >= t['tp']) or \
                   (t['type'] == 'SHORT' and l <= t['tp']):
                    exit_p = t['tp'] * (1 + SLIPPAGE if t['type'] == 'SHORT' else 1 - SLIPPAGE)
                    self._record(t, exit_p, 'TP', ts)
                    open_trade = None
                    continue

            if open_trade:
                continue   # Only one trade at a time

            # ---- Entry logic ----
            # HTF bias
            try:
                htf_idx = df_1h.index.get_indexer([ts], method='pad')[0]
                if htf_idx < 0:
                    continue
                bias = htf_bias(df_1h.iloc[:htf_idx + 1])
            except Exception:
                continue

            window    = df.iloc[max(0, i - 60):i + 1]
            bos_up, bos_down     = break_of_structure(window)
            sw_high, sw_low      = liquidity_sweep(window)
            gap_up, gap_down     = fvg(window)
            bull_ob, bear_ob     = order_block(window)
            long_s, short_s      = confluence_score(
                bias, bos_up, bos_down, sw_high, sw_low,
                gap_up, gap_down, bull_ob, bear_ob, price,
            )
            rsi = df['RSI'].iloc[i]

            # Long setup
            if long_s >= 2 and rsi < 65 and bias == 'BULL':
                sl_p = price - self.sl_mult * atr
                tp_p = price + self.tp_mult * atr
                rr   = (tp_p - price) / (price - sl_p + 1e-10)
                if rr >= MIN_RR:
                    entry_p = price * (1 + SLIPPAGE)
                    open_trade = {
                        'type': 'LONG', 'entry': entry_p,
                        'tp': tp_p, 'sl': sl_p, 'initial_sl': sl_p,
                        'atr': atr, 'size': 1.0,
                        'open_ts': ts, 'partial': False,
                    }

            # Short setup
            elif short_s >= 2 and rsi > 35 and bias == 'BEAR':
                sl_p = price + self.sl_mult * atr
                tp_p = price - self.tp_mult * atr
                rr   = (price - tp_p) / (sl_p - price + 1e-10)
                if rr >= MIN_RR:
                    entry_p = price * (1 - SLIPPAGE)
                    open_trade = {
                        'type': 'SHORT', 'entry': entry_p,
                        'tp': tp_p, 'sl': sl_p, 'initial_sl': sl_p,
                        'atr': atr, 'size': 1.0,
                        'open_ts': ts, 'partial': False,
                    }

            # Progress indicator (every 1000 bars)
            if i % 1000 == 0:
                progress = (i - 200) / (n - LOOKAHEAD - 200) * 100
                logger.info(f"  Progress: {progress:.1f}% | Trades: {len(self.trades)} | Equity: {self.balance:.3f}x")

        return self.trades, np.array(self.equity)

    def _record(self, trade, exit_price, exit_reason, ts):
        if trade['type'] == 'LONG':
            raw_r = (exit_price - trade['entry']) / (trade['entry'] - trade['initial_sl'] + 1e-10)
        else:
            raw_r = (trade['entry'] - exit_price) / (trade['initial_sl'] - trade['entry'] + 1e-10)

        fee_cost = FEE_MAKER * 2
        net_r    = raw_r * trade['size'] - fee_cost

        self.balance *= (1 + net_r * RISK_PERCENT)
        self.equity.append(self.balance)

        self.trades.append({
            'symbol':      self.symbol,
            'type':        trade['type'],
            'entry':       round(trade['entry'], 2),
            'exit':        round(exit_price, 2),
            'exit_reason': exit_reason,
            'r_multiple':  round(net_r, 3),
            'open_ts':     trade['open_ts'],
            'close_ts':    ts,
        })


# ==================== RESULTS ====================

def save_results(symbol, trades, equity):
    os.makedirs(LOG_DIR, exist_ok=True)

    trade_file  = os.path.join(LOG_DIR, f'backtest_trades_{symbol.replace("/","")}.csv')
    equity_file = os.path.join(LOG_DIR, f'backtest_equity_{symbol.replace("/","")}.csv')

    if trades:
        keys = trades[0].keys()
        with open(trade_file, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(trades)

    with open(equity_file, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bar', 'equity'])
        for i, e in enumerate(equity):
            w.writerow([i, round(e, 6)])

    logger.info(f"  Saved: {trade_file}")
    logger.info(f"  Saved: {equity_file}")


def print_summary(symbol, trades, equity):
    if not trades:
        print(f"\n{symbol}: No trades generated.")
        return

    returns = np.array([t['r_multiple'] for t in trades])
    stats   = trade_stats(returns)
    wins    = [t for t in trades if t['r_multiple'] > 0]
    losses  = [t for t in trades if t['r_multiple'] <= 0]
    final   = equity[-1]

    print(f"\n{'='*60}")
    print(f"  {symbol} Backtest Results ({len(equity)} candles)")
    print(f"{'='*60}")
    print(f"  Total trades       : {stats['trades']}")
    print(f"  Win rate           : {stats['win_rate']*100:.1f}%")
    print(f"  Avg R per trade    : {stats['avg_r']:+.3f}R")
    print(f"  Sharpe (annualised): {stats['sharpe']:+.2f}")
    print(f"  Max drawdown       : {stats['max_dd']*100:.1f}%")
    print(f"  Final equity       : {final:.3f}x")
    print(f"  Total return       : {(final-1)*100:+.1f}%")
    print(f"  Avg win            : {np.mean([t['r_multiple'] for t in wins]):.3f}R" if wins else "  Avg win            : —")
    print(f"  Avg loss           : {np.mean([t['r_multiple'] for t in losses]):.3f}R" if losses else "  Avg loss           : —")
    
    if trades:
        first_trade = trades[0]['open_ts']
        last_trade = trades[-1]['close_ts']
        duration = (last_trade - first_trade).days
        print(f"  Trading period     : {duration} days")
    
    print(f"{'='*60}")


# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  BACKTEST - Using Real Historical Data from Binance")
    print("="*60)
    print(f"  Fetching {BACKTEST_MONTHS} months of data")
    print(f"  Timeframes: {TF_SIGNAL} (signal), {TF_BIAS} (bias)")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print("="*60 + "\n")

    exchange = connect_production()

    for sym in SYMBOLS:
        print(f"\n{'*'*60}")
        print(f"  Processing {sym}")
        print(f"{'*'*60}\n")

        # Fetch historical data with caching
        df_15m = get_cached_or_fetch(exchange, sym, TF_SIGNAL, months=BACKTEST_MONTHS)
        df_1h  = get_cached_or_fetch(exchange, sym, TF_BIAS, months=BACKTEST_MONTHS)

        # Fetch OI and funding (will be limited by API)
        oi = fetch_open_interest_historical(exchange, sym, tf=TF_SIGNAL, months=3)
        
        try:
            fr = exchange.fetch_funding_rate(sym)
            funding_rate = float(fr.get('fundingRate', 0.0))
        except:
            funding_rate = 0.0
            logger.warning("Funding rate unavailable, using 0.0")

        # Build features
        logger.info("Building features for 15m data...")
        df_15m = build_features(df_15m, funding_rate=funding_rate, oi_df=oi)
        
        logger.info("Building features for 1h data...")
        df_1h = build_features(df_1h, funding_rate=funding_rate)

        # Align 1h to 15m index
        df_1h = df_1h.reindex(df_15m.index, method='ffill').dropna()

        logger.info(f"Data prepared: {len(df_15m):,} candles")

        # Optimize multipliers
        logger.info("Optimizing TP/SL multipliers...")
        tp_mult, sl_mult = optimize_multipliers(df_15m)
        logger.info(f"  Optimal TP: {tp_mult:.2f}x ATR | SL: {sl_mult:.2f}x ATR")

        # Run backtest
        bt = Backtester(sym, df_15m, df_1h, tp_mult, sl_mult)
        trades, equity = bt.run()

        # Results
        print_summary(sym, trades, equity)
        save_results(sym, trades, equity)

    print("\n" + "="*60)
    print("  BACKTEST COMPLETE")
    print("="*60)
    print(f"  Results saved in: {LOG_DIR}/")
    print(f"  Cached data in: {CACHE_DIR}/")
    print("="*60 + "\n")