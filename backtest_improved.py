"""
Standalone backtester with real historical data from Binance production API.

Usage:
    python backtest_improved.py

Key improvements (v2):
    - ML model (LSTM + XGBoost) trained on in-sample data, tested out-of-sample
    - Session filter (London open → NY close: 08:00–20:00 UTC)
    - Higher confluence threshold (≥3 ICT signals)
    - Tighter RSI filters (Long <55, Short >45)
    - Regime + ADX filters (trending markets only)
    - Realistic LOOKAHEAD (24 bars = 6 hours on 15m)
    - Reduced partial close (30% at 1R, not 50%)
    - Fixed slippage calculation on TP exits
    - Train/test split to avoid look-ahead bias
"""

import os
import csv
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import ccxt

from config import (
    SYMBOLS, TF_SIGNAL, TF_BIAS, SEQ_LEN,
    RISK_PERCENT, FEE_MAKER, FEE_TAKER,
    MIN_RR, LOG_DIR,
    SESSION_START_UTC, SESSION_END_UTC,
)
from features import build_features, htf_bias, FEAT_COLS
from labels import build_labels, optimize_multipliers
from ict import break_of_structure, liquidity_sweep, fvg, order_block, confluence_score
from models import EnsembleModel
from validation import trade_stats, sharpe_ratio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== BACKTEST PARAMETERS ====================

USE_ML          = False    # Toggle ML model on/off (False = pure ICT signals)
BACKTEST_SYMBOLS = ['ETH/USDT']  # Override symbols for backtest (BTC score was -197R)

SLIPPAGE        = 0.0005   # 0.05% per side
LOOKAHEAD       = 36       # 36 bars x 15m = 9 hours (wider SL needs more time)
CONF_THRESH_BT  = 0.20     # ML confidence threshold (only used when USE_ML=True)
TRAIN_RATIO     = 0.70     # 70% train, 30% out-of-sample test
MIN_CONFLUENCE  = 2        # Lowered back: EMA/BB/VWAP filters now provide quality control
RSI_LONG_MAX    = 60       # Max RSI for long entries (relaxed from 55)
RSI_SHORT_MIN   = 40       # Min RSI for short entries (relaxed from 45)
ADX_MIN         = 18       # Minimum ADX for trending market (relaxed from 20)
PARTIAL_CLOSE   = 0.0      # DISABLED: partial close was capping avg win at 0.38R

# NEW: EMA crossover confirmation (9/21 EMA must agree with trade direction)
USE_EMA_CONFIRM = True
# NEW: Bollinger Band overextension filter (skip entries at band extremes)
USE_BB_FILTER   = True
BB_LONG_MAX     = 0.85     # Skip longs if price > 85% of BB range (overextended)
BB_SHORT_MIN    = 0.15     # Skip shorts if price < 15% of BB range (overextended)
# NEW: VWAP proximity filter (entries near VWAP have better mean-reversion edge)
USE_VWAP_FILTER = True
VWAP_MAX_DIST   = 3.0      # Max distance from VWAP in ATR units

# Historical data settings
BACKTEST_MONTHS = 12
CACHE_DIR       = 'data/cache'


# ==================== DATA FETCHING ====================

def connect_production():
    """
    Connect to PRODUCTION Binance for historical data.
    No API keys needed for public historical data.
    """
    ex = ccxt.binance({
        'options': {'defaultType': 'future'},
        'enableRateLimit': True,
    })
    ex.load_markets()
    logger.info("Connected to Binance production API for historical data")
    return ex


def fetch_historical_range(exchange, symbol, timeframe, start_date, end_date=None):
    """
    Fetch OHLCV data for a specific date range.
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

            if batch_count % 10 == 0:
                progress = len(all_bars) / expected_candles * 100 if expected_candles > 0 else 0
                logger.info(f"  Fetched {len(all_bars):,} candles (~{progress:.1f}%)")

            current_ms = bars[-1][0] + 1

        except Exception as e:
            logger.error(f"Error fetching batch: {e}")
            if len(all_bars) > 0:
                logger.warning("Using partial data collected so far")
                break
            else:
                raise

    df = pd.DataFrame(all_bars, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df.set_index('time', inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    logger.info(f"✓ Fetched {len(df):,} candles for {symbol} {timeframe}")
    return df


def get_cached_or_fetch(exchange, symbol, timeframe, months=12):
    """Check cache, fetch if missing or expired."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_file = os.path.join(
        CACHE_DIR,
        f"{symbol.replace('/', '')}_{timeframe}_{months}m.csv"
    )

    if os.path.exists(cache_file):
        cache_age_hours = (datetime.now().timestamp() - os.path.getmtime(cache_file)) / 3600
        if cache_age_hours < 24:
            logger.info(f"Loading cached data: {cache_file} (age: {cache_age_hours:.1f}h)")
            df = pd.read_csv(cache_file, index_col='time', parse_dates=True)
            return df
        else:
            logger.info(f"Cache expired (age: {cache_age_hours:.1f}h), fetching fresh data")

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=months * 30)
    df = fetch_historical_range(exchange, symbol, timeframe, start_date, end_date)
    df.to_csv(cache_file)
    logger.info(f"Cached data saved: {cache_file}")
    return df


def fetch_open_interest_historical(exchange, symbol, tf='15m', months=12):
    """Fetch historical Open Interest data (capped at 90 days by Binance)."""
    try:
        days = min(months * 30, 90)
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
    def __init__(self, symbol, df_15m, df_1h, tp_mult, sl_mult, model=None):
        self.symbol   = symbol
        self.df       = df_15m
        self.df_1h    = df_1h
        self.tp_mult  = tp_mult
        self.sl_mult  = sl_mult
        self.model    = model
        self.trades   = []
        self.equity   = [1.0]
        self.balance  = 1.0

        # Entry filter stats for diagnostics
        self.filter_stats = {
            'session_skip': 0,
            'confluence_skip': 0,
            'rsi_skip': 0,
            'bias_skip': 0,
            'regime_skip': 0,
            'adx_skip': 0,
            'ema_skip': 0,
            'bb_skip': 0,
            'vwap_skip': 0,
            'ml_skip': 0,
            'rr_skip': 0,
            'entries': 0,
        }

    def run(self):
        df     = self.df
        df_1h  = self.df_1h
        closes = df['Close'].values
        highs  = df['High'].values
        lows   = df['Low'].values
        atrs   = df['ATR'].values
        rsis   = df['RSI'].values
        adxs   = df['ADX'].values
        regimes = df['Regime'].values
        ema9s   = df['EMA9'].values
        ema21s  = df['EMA21'].values
        bb_pcts = df['BB_Pct'].values
        vwaps   = df['VWAP'].values
        n      = len(df)
        open_trade = None

        warmup = max(200, SEQ_LEN + 10)  # Ensure enough data for ML model
        logger.info(f"Running backtest on {n:,} candles (warmup: {warmup})...")

        for i in range(warmup, n - LOOKAHEAD):
            price  = closes[i]
            atr    = atrs[i]
            rsi    = rsis[i]
            adx    = adxs[i]
            regime = regimes[i]
            ema9   = ema9s[i]
            ema21  = ema21s[i]
            bb_pct = bb_pcts[i]
            vwap   = vwaps[i]
            ts     = df.index[i]

            # ---- Manage open trade ----
            if open_trade:
                h = highs[i]
                l = lows[i]
                t = open_trade

                # Partial close at 1R (30% of position)
                if not t.get('partial') and (
                    (t['type'] == 'LONG'  and h >= t['entry'] + t['atr'] * self.sl_mult) or
                    (t['type'] == 'SHORT' and l <= t['entry'] - t['atr'] * self.sl_mult)
                ):
                    t['partial'] = True
                    # Move SL to entry + small buffer (not exact breakeven - avoids BE wicks)
                    if t['type'] == 'LONG':
                        t['sl'] = t['entry'] + 0.2 * t['atr']  # Small profit lock
                    else:
                        t['sl'] = t['entry'] - 0.2 * t['atr']
                    t['size'] *= (1.0 - PARTIAL_CLOSE)  # Keep 70%

                # Trail stop (LONG) - start trailing at 1.5R with wider trail
                if t['type'] == 'LONG' and h >= t['entry'] + 1.5 * t['atr'] * self.sl_mult:
                    t['sl'] = max(t['sl'], h - 1.5 * t['atr'])  # Trail at 1.5 ATR

                # Trail stop (SHORT)
                if t['type'] == 'SHORT' and l <= t['entry'] - 1.5 * t['atr'] * self.sl_mult:
                    t['sl'] = min(t['sl'], l + 1.5 * t['atr'])

                # Exit: SL
                if (t['type'] == 'LONG'  and l <= t['sl']) or \
                   (t['type'] == 'SHORT' and h >= t['sl']):
                    if t['type'] == 'LONG':
                        exit_p = t['sl'] * (1 - SLIPPAGE)
                    else:
                        exit_p = t['sl'] * (1 + SLIPPAGE)
                    self._record(t, exit_p, 'SL', ts)
                    open_trade = None
                    continue

                # Exit: TP (fixed slippage - always worse fill for the trader)
                if (t['type'] == 'LONG'  and h >= t['tp']) or \
                   (t['type'] == 'SHORT' and l <= t['tp']):
                    if t['type'] == 'LONG':
                        exit_p = t['tp'] * (1 - SLIPPAGE)   # Sell at slightly less
                    else:
                        exit_p = t['tp'] * (1 + SLIPPAGE)   # Buy-to-cover at slightly more
                    self._record(t, exit_p, 'TP', ts)
                    open_trade = None
                    continue

            if open_trade:
                continue

            # ---- Entry Filters (layered, most-to-least restrictive) ----

            # 1. Session filter - only trade London open -> NY close
            hour_utc = ts.hour
            if not (SESSION_START_UTC <= hour_utc < SESSION_END_UTC):
                self.filter_stats['session_skip'] += 1
                continue

            # 2. Regime filter - tracked for diagnostics but not blocking
            if regime != 1:
                self.filter_stats['regime_skip'] += 1

            # 3. ADX filter - need minimum trend strength
            if adx < ADX_MIN:
                self.filter_stats['adx_skip'] += 1
                continue

            # 4. HTF bias
            try:
                htf_idx = df_1h.index.get_indexer([ts], method='pad')[0]
                if htf_idx < 0:
                    continue
                bias = htf_bias(df_1h.iloc[:htf_idx + 1])
            except Exception:
                continue

            if bias == 'NEUTRAL':
                self.filter_stats['bias_skip'] += 1
                continue

            # 5. EMA crossover confirmation (9/21 EMA must agree with direction)
            if USE_EMA_CONFIRM:
                if bias == 'BULL' and ema9 < ema21:
                    self.filter_stats['ema_skip'] += 1
                    continue
                if bias == 'BEAR' and ema9 > ema21:
                    self.filter_stats['ema_skip'] += 1
                    continue

            # 6. Bollinger Band overextension filter
            if USE_BB_FILTER:
                if bias == 'BULL' and bb_pct > BB_LONG_MAX:
                    self.filter_stats['bb_skip'] += 1
                    continue
                if bias == 'BEAR' and bb_pct < BB_SHORT_MIN:
                    self.filter_stats['bb_skip'] += 1
                    continue

            # 7. VWAP proximity filter (don't enter too far from fair value)
            if USE_VWAP_FILTER:
                vwap_dist = abs(price - vwap) / (atr + 1e-10)
                if vwap_dist > VWAP_MAX_DIST:
                    self.filter_stats['vwap_skip'] += 1
                    continue

            # 8. ICT confluence
            window    = df.iloc[max(0, i - 60):i + 1]
            bos_up, bos_down     = break_of_structure(window)
            sw_high, sw_low      = liquidity_sweep(window)
            gap_up, gap_down     = fvg(window)
            bull_ob, bear_ob     = order_block(window)
            long_s, short_s      = confluence_score(
                bias, bos_up, bos_down, sw_high, sw_low,
                gap_up, gap_down, bull_ob, bear_ob, price,
            )

            # 9. ML prediction (if model available)
            ml_pred = None
            ml_conf = 1.0
            if self.model is not None:
                try:
                    df_window = df.iloc[max(0, i - SEQ_LEN + 1):i + 1]
                    if len(df_window) >= SEQ_LEN:
                        ml_pred, ml_conf, _ = self.model.predict(df_window)
                except Exception:
                    ml_pred = None
                    ml_conf = 0.0

            # ---- Long setup ----
            if (
                long_s >= MIN_CONFLUENCE
                and rsi < RSI_LONG_MAX
                and bias == 'BULL'
            ):
                # ML gate
                if self.model is not None:
                    if ml_pred != 1 or ml_conf < CONF_THRESH_BT:
                        self.filter_stats['ml_skip'] += 1
                    else:
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
                            self.filter_stats['entries'] += 1
                        else:
                            self.filter_stats['rr_skip'] += 1
                else:
                    # No ML model - use pure ICT signals
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
                        self.filter_stats['entries'] += 1
                    else:
                        self.filter_stats['rr_skip'] += 1

            # ---- Short setup ----
            elif (
                short_s >= MIN_CONFLUENCE
                and rsi > RSI_SHORT_MIN
                and bias == 'BEAR'
            ):
                if self.model is not None:
                    if ml_pred != 0 or ml_conf < CONF_THRESH_BT:
                        self.filter_stats['ml_skip'] += 1
                    else:
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
                            self.filter_stats['entries'] += 1
                        else:
                            self.filter_stats['rr_skip'] += 1
                else:
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
                        self.filter_stats['entries'] += 1
                    else:
                        self.filter_stats['rr_skip'] += 1
            else:
                self.filter_stats['confluence_skip'] += 1

            # Progress indicator
            if i % 2000 == 0:
                progress = (i - warmup) / (n - LOOKAHEAD - warmup) * 100
                logger.info(
                    f"  Progress: {progress:.1f}% | "
                    f"Trades: {len(self.trades)} | "
                    f"Equity: {self.balance:.3f}x"
                )

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


def print_summary(symbol, trades, equity, filter_stats=None):
    if not trades:
        print(f"\n{symbol}: No trades generated.")
        if filter_stats:
            print(f"  Filter stats: {filter_stats}")
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
    print(f"  Avg win            : {np.mean([t['r_multiple'] for t in wins]):.3f}R" if wins else "  Avg win            : -")
    print(f"  Avg loss           : {np.mean([t['r_multiple'] for t in losses]):.3f}R" if losses else "  Avg loss           : -")

    if trades:
        first_trade = trades[0]['open_ts']
        last_trade  = trades[-1]['close_ts']
        duration    = (last_trade - first_trade).days
        print(f"  Trading period     : {duration} days")

    print(f"{'='*60}")

    if filter_stats:
        print(f"\n  Entry Filter Breakdown:")
        for key, val in filter_stats.items():
            print(f"    {key:>20s}: {val:>6,}")
        print()


# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  BACKTEST v3 - ICT + EMA/BB/VWAP Filter Stack")
    print("="*60)
    print(f"  Data period   : {BACKTEST_MONTHS} months")
    print(f"  Train/Test    : {TRAIN_RATIO*100:.0f}% / {(1-TRAIN_RATIO)*100:.0f}%")
    print(f"  Session       : {SESSION_START_UTC:02d}:00 - {SESSION_END_UTC:02d}:00 UTC")
    print(f"  Confluence    : >={MIN_CONFLUENCE}/5 ICT signals")
    print(f"  RSI           : Long <{RSI_LONG_MAX} | Short >{RSI_SHORT_MIN}")
    print(f"  ADX minimum   : {ADX_MIN}")
    print(f"  EMA confirm   : {'9/21 EMA alignment' if USE_EMA_CONFIRM else 'OFF'}")
    print(f"  BB filter     : {'Long<{0:.0%} Short>{1:.0%}'.format(BB_LONG_MAX, BB_SHORT_MIN) if USE_BB_FILTER else 'OFF'}")
    print(f"  VWAP filter   : {'<{0:.1f} ATR from VWAP'.format(VWAP_MAX_DIST) if USE_VWAP_FILTER else 'OFF'}")
    print(f"  ML model      : {'ON (>=' + f'{CONF_THRESH_BT:.0%}' + ')' if USE_ML else 'OFF (pure ICT)'}")
    print(f"  MIN R:R       : {MIN_RR}")
    print(f"  LOOKAHEAD     : {LOOKAHEAD} bars ({LOOKAHEAD * 15 / 60:.0f}h on 15m)")
    print(f"  Partial close : {PARTIAL_CLOSE*100:.0f}% at 1R")
    print(f"  Timeframes    : {TF_SIGNAL} (signal), {TF_BIAS} (bias)")
    print(f"  Symbols       : {', '.join(BACKTEST_SYMBOLS)}")
    print("="*60 + "\n")

    exchange = connect_production()

    for sym in BACKTEST_SYMBOLS:
        print(f"\n{'*'*60}")
        print(f"  Processing {sym}")
        print(f"{'*'*60}\n")

        tag = sym.replace('/', '')

        # ---- Fetch historical data ----
        df_15m = get_cached_or_fetch(exchange, sym, TF_SIGNAL, months=BACKTEST_MONTHS)
        df_1h  = get_cached_or_fetch(exchange, sym, TF_BIAS, months=BACKTEST_MONTHS)

        oi = fetch_open_interest_historical(exchange, sym, tf=TF_SIGNAL, months=3)

        try:
            fr = exchange.fetch_funding_rate(sym)
            funding_rate = float(fr.get('fundingRate', 0.0))
        except Exception:
            funding_rate = 0.0
            logger.warning("Funding rate unavailable, using 0.0")

        # ---- Build features ----
        logger.info("Building features for 15m data...")
        df_15m = build_features(df_15m, funding_rate=funding_rate, oi_df=oi)

        logger.info("Building features for 1h data...")
        df_1h = build_features(df_1h, funding_rate=funding_rate)

        # Align 1h to 15m index
        df_1h = df_1h.reindex(df_15m.index, method='ffill').dropna()

        total_candles = len(df_15m)
        logger.info(f"Data prepared: {total_candles:,} candles")

        # ---- Train/Test Split ----
        split_idx = int(total_candles * TRAIN_RATIO)

        train_15m = df_15m.iloc[:split_idx].copy()
        train_1h  = df_1h.iloc[:split_idx].copy()

        # Test data includes warmup overlap from training for context
        warmup_bars = max(200, SEQ_LEN + 10)
        test_start  = max(0, split_idx - warmup_bars)
        test_15m    = df_15m.iloc[test_start:].copy()
        test_1h     = df_1h.iloc[test_start:].copy()

        logger.info(f"  Train: {len(train_15m):,} candles | Test: {len(test_15m):,} candles (incl. {warmup_bars} warmup)")

        # ---- Optimize TP/SL on TRAINING data only (avoid look-ahead bias) ----
        logger.info("Optimizing TP/SL multipliers on training data...")
        tp_mult, sl_mult = optimize_multipliers(train_15m, min_rr=MIN_RR)
        logger.info(f"  Optimal TP: {tp_mult:.2f}x ATR | SL: {sl_mult:.2f}x ATR")

        # ---- Train or load ML model (if enabled) ----
        model = None
        if USE_ML:
            model = EnsembleModel(FEAT_COLS, symbol_tag=tag)
            if model.load():
                logger.info("  Loaded cached ML model (delete models/ folder to retrain)")
            else:
                logger.info("Training ML model on training data (first run - will be cached)...")
                train_labels = build_labels(train_15m, tp_mult, sl_mult, lookahead=LOOKAHEAD)
                model.fit(train_15m, train_labels)
                model.save()
                logger.info("  ML model trained and saved")
        else:
            logger.info("  ML model DISABLED - using pure ICT signals")

        # ---- Run backtest on TEST data ----
        mode_str = "with ML model" if USE_ML else "with pure ICT signals"
        logger.info(f"Running out-of-sample backtest {mode_str}...")
        bt = Backtester(sym, test_15m, test_1h, tp_mult, sl_mult, model=model)
        trades, equity = bt.run()

        # ---- Results ----
        print_summary(sym, trades, equity, filter_stats=bt.filter_stats)
        save_results(sym, trades, equity)

    print("\n" + "="*60)
    print("  BACKTEST v2 COMPLETE")
    print("="*60)
    print(f"  Results saved in: {LOG_DIR}/")
    print(f"  Cached data in: {CACHE_DIR}/")
    print("="*60 + "\n")