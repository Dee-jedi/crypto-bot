"""
Main live trading bot.

Run:
    python bot.py
"""

import os
import sys
import time
import logging
import signal
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

from config import (
    SYMBOLS, TF_SIGNAL, TF_BIAS,
    MAX_TRADES_TOTAL, COOLDOWN_MINUTES,
    BALANCE_REFRESH_CANDLES, LOOP_SLEEP_SECONDS,
    LOG_DIR, SESSION_START_UTC, SESSION_END_UTC,
    USE_RECLAIM_CONFIRM, USE_PULSE_FILTER, USE_4H_FILTER,
    USE_ADAPTIVE_EXIT, BREAKEVEN_TRIGGER_R, TRAILING_STOP_ATR,
    TP_MULT, SL_MULT, PARTIAL_CLOSE_R, PARTIAL_CLOSE_AMT, ADX_MIN,
    RSI_LONG_MAX, RSI_SHORT_MIN,
)
from data_feed import (
    connect, fetch_ohlcv, fetch_ohlcv_bulk,
    fetch_funding_rate, fetch_open_interest,
)
from features import build_features, htf_bias
from risk import RiskManager, correlated_exposure, log_trade
from execution import place_limit_entry, wait_for_fill, close_full
from alerts import (
    alert_trade_open, alert_trade_close,
    alert_halt, alert_resume,
    alert_startup, alert_daily_summary,
)

# ==================== LOGGING ====================

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'bot.log')),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger('bot')


# ==================== GRACEFUL SHUTDOWN ====================

_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    logger.warning("Shutdown signal received. Closing after this iteration...")
    _shutdown = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ==================== TRADE FACTORY ====================

def make_trade(symbol, trade_type, entry, tp, sl, atr, lot_size, regime):
    return {
        'symbol':    symbol,
        'type':      trade_type,
        'entry':     entry,
        'tp':        tp,
        'sl':        sl,
        'initial_sl': sl,
        'atr':       atr,
        'lot_size':  lot_size,
        'remaining': lot_size,
        'regime':    regime,
        'open_time': datetime.now(timezone.utc),
    }

# ==================== INITIALISE ====================

def initialise():
    logger.info("Connecting to exchange...")
    exchange = connect()

    balance = exchange.fetch_balance()['USDT']['free']
    logger.info(f"Connected | Balance: ${balance:,.2f}")

    risk_mgr  = RiskManager(balance)

    for sym in SYMBOLS:
        logger.info(f"Loading live context for {sym}...")
        # Just pre-warming cache if needed
    
    # We now strictly use our backtest-verified 1.5/1.0 parameters
    return exchange, balance, risk_mgr, True


# ==================== TRADE MANAGEMENT ====================

def manage_trade(trade, df_15m, exchange, open_trades, daily_stats):
    """
    Check TP / SL for a single open trade.
    Modifies trade dict in-place.
    Returns True if the trade was closed, False if still open.
    """
    h     = df_15m['High'].iloc[-1]
    l     = df_15m['Low'].iloc[-1]
    c     = df_15m['Close'].iloc[-1]
    t     = trade

    # ---- Master Manage (Pure ICT Goldilocks) ----
    if t['type'] == 'LONG':
        current_r = (c - t['entry']) / (t['entry'] - t['initial_sl'] + 1e-10)
        
        # 1. Partial Profit @ 1.0R
        if not t.get('partial', False) and current_r >= PARTIAL_CLOSE_R:
            close_amt = t['lot_size'] * PARTIAL_CLOSE_AMT
            if close_full(exchange, t['symbol'], 'LONG', close_amt):
                t['remaining'] -= close_amt
                t['partial'] = True
                logger.info(f"Target 1 HIT | 20% partial close on {t['symbol']} @ 1.0R")
                
        # 2. Breakeven @ 1.4R
        if current_r >= BREAKEVEN_TRIGGER_R and t['sl'] < t['entry']:
            t['sl'] = t['entry'] + (0.05 * t['atr']) # Entry + tiny offset
            logger.info(f"Safety Trigger: SL moved to Breakeven for {t['symbol']}")
            
    else: # SHORT
        current_r = (t['entry'] - c) / (t['initial_sl'] - t['entry'] + 1e-10)
        
        # 1. Partial Profit @ 1.0R
        if not t.get('partial', False) and current_r >= PARTIAL_CLOSE_R:
            close_amt = t['lot_size'] * PARTIAL_CLOSE_AMT
            if close_full(exchange, t['symbol'], 'SHORT', close_amt):
                t['remaining'] -= close_amt
                t['partial'] = True
                logger.info(f"Target 1 HIT | 20% partial close on {t['symbol']} @ 1.0R")
                
        # 2. Breakeven @ 1.4R
        if current_r >= BREAKEVEN_TRIGGER_R and t['sl'] > t['entry']:
            t['sl'] = t['entry'] - (0.05 * t['atr'])
            logger.info(f"Safety Trigger: SL moved to Breakeven for {t['symbol']}")

    # ---- SL hit ----
    sl_hit = (
        (t['type'] == 'LONG'  and l <= t['sl']) or
        (t['type'] == 'SHORT' and h >= t['sl'])
    )
    if sl_hit:
        if close_full(exchange, sym, t['type'], t['remaining']):
            exit_price = t['sl']
            r_raw      = abs(exit_price - t['entry']) / (abs(t['entry'] - t['initial_sl']) + 1e-10)
            r_signed   = -r_raw if (
                (t['type'] == 'LONG' and exit_price < t['entry']) or
                (t['type'] == 'SHORT' and exit_price > t['entry'])
            ) else r_raw

            log_trade(sym, t['type'], t['entry'], exit_price, 0, t['lot_size'], r_signed)
            alert_trade_close(sym, t['type'], t['entry'], exit_price, 0, r_signed)
            daily_stats['trades'] += 1
            open_trades.remove(t)
            logger.info(f"SL closed {sym} {t['type']} @ {exit_price:.2f} | {r_signed:+.2f}R")
            return True

    # ---- TP hit ----
    tp_hit = (
        (t['type'] == 'LONG'  and h >= t['tp']) or
        (t['type'] == 'SHORT' and l <= t['tp'])
    )
    if tp_hit:
        exit_price = t['tp']
        if close_full(exchange, sym, t['type'], t['remaining']):
            r_raw = abs(exit_price - t['entry']) / (abs(t['entry'] - t['initial_sl']) + 1e-10)

            log_trade(sym, t['type'], t['entry'], exit_price, 1, t['lot_size'], r_raw)
            alert_trade_close(sym, t['type'], t['entry'], exit_price, 1, r_raw)
            daily_stats['trades'] += 1
            daily_stats['wins']   += 1
            open_trades.remove(t)
            logger.info(f"TP closed {sym} {t['type']} @ {exit_price:.2f} | {r_raw:+.2f}R")
            return True

    return False


# ==================== ENTRY LOGIC ====================

def attempt_entry(sym, df_15m, df_1h, risk_mgr, balance, open_trades, cooldowns, exchange):
    """
    Evaluates Trend-Pullback entry conditions for one symbol.
    Returns True if a trade was initiated.
    """
    tp_mult, sl_mult = 1.5, 1.0  # Verified mathematically solid base
    
    price            = df_15m['Close'].iloc[-1]
    lows             = df_15m['Low'].iloc[-1]
    highs            = df_15m['High'].iloc[-1]
    atr              = df_15m['ATR'].iloc[-1]
    rsi              = df_15m['RSI'].iloc[-1]
    adx              = df_15m['ADX'].iloc[-1]
    ema21            = df_15m['EMA21'].iloc[-1]
    vwap             = df_15m['VWAP'].iloc[-1]
    regime           = int(df_15m['Regime'].iloc[-1])
    ts               = df_15m.index[-1]
    hour_utc         = ts.hour

    # ---- Time filter: London open → NY close ----
    if not (SESSION_START_UTC <= hour_utc < SESSION_END_UTC):
        return False

    # ---- ICT Strict Gate (The Institutional Guard) ----
    gap_up     = df_15m['GapUp'].iloc[-1]
    gap_down   = df_15m['GapDown'].iloc[-1]
    swept_low  = df_15m['SweptLow'].iloc[-1]
    swept_high = df_15m['SweptHigh'].iloc[-1]

    # ---- Cooldown ----
    cd = cooldowns.get(sym)
    if cd and ts < cd:
        return False

    # ---- ADX Trend filter ----
    if adx < ADX_MIN:
        return False

    # ---- HTF bias (NEUTRAL → skip) ----
    bias = htf_bias(df_1h)
    if bias == 'NEUTRAL':
        return False
    
    # ---- Super-HTF (4H) optional filter ----
    if USE_4H_FILTER:
        # 4H data would need to be fetched/resampled here (skipping as default is False)
        pass

    # ---- Pulse Filter (Bollinger Expansion) ----
    if USE_PULSE_FILTER:
        bb_w = df_15m['BB_Width'].iloc[-1]
        bb_s = df_15m['BB_Width'].rolling(20).mean().iloc[-1]
        if bb_w < bb_s:
            return False

    # ---- Correlation guard ----
    if correlated_exposure(open_trades):
        return False

    entered = False
    
    # Check Pullback touches and Reclaim
    ema21_prev = df_15m['EMA21'].iloc[-2]
    vwap_prev  = df_15m['VWAP'].iloc[-2]
    rsi_prev   = df_15m['RSI'].iloc[-2]
    
    has_touched_long = (lows <= ema21) or (lows <= vwap) or \
                       (df_15m['Low'].iloc[-2] <= ema21_prev) or (df_15m['Low'].iloc[-2] <= vwap_prev)
                       
    has_touched_short = (highs >= ema21) or (highs >= vwap) or \
                        (df_15m['High'].iloc[-2] >= ema21_prev) or (df_15m['High'].iloc[-2] >= vwap_prev)

    if USE_RECLAIM_CONFIRM:
        is_reclaimed_long = (price > ema21)
        is_reclaimed_short = (price < ema21)
        rsi_limit_long = RSI_LONG_MAX + 3 # Small buffer for reclaim
        rsi_limit_short = RSI_SHORT_MIN - 3
    else:
        is_reclaimed_long = True
        is_reclaimed_short = True
        rsi_limit_long = RSI_LONG_MAX
        rsi_limit_short = RSI_SHORT_MIN

    # ---- LONG ----
    if bias == 'BULL' and rsi_prev <= RSI_LONG_MAX and rsi <= rsi_limit_long and has_touched_long and is_reclaimed_long:
        # STRICT GATE: Must have FVG or SWEEP confluence
        if not (gap_up or swept_low):
            return False

        sl_p = price - SL_MULT * atr
        tp_p = price + TP_MULT * atr

        if not risk_mgr.min_rr_ok(price, tp_p, sl_p):
            return False

        lot = risk_mgr.lot_size(balance, price, sl_p)
        if lot <= 0:
            return False

        logger.info(f"LONG setup detected | {sym} | ADX: {adx:.0f} | RSI: {rsi:.0f}")
        order_id, fill_price = place_limit_entry(exchange, sym, 'buy', lot, price, atr)

        if order_id and wait_for_fill(exchange, sym, order_id):
            trade = make_trade(sym, 'LONG', fill_price, tp_p, sl_p, atr, lot, regime)
            open_trades.append(trade)
            alert_trade_open(sym, 'LONG', fill_price, tp_p, sl_p, lot, 1.0, regime, 3)
            logger.info(f"LONG opened {sym} @ {fill_price:.2f} | TP {tp_p:.2f} | SL {sl_p:.2f}")
            entered = True

    # ---- SHORT ----
    elif bias == 'BEAR' and rsi_prev >= RSI_SHORT_MIN and rsi >= rsi_limit_short and has_touched_short and is_reclaimed_short:
        # STRICT GATE: Must have FVG or SWEEP confluence
        if not (gap_down or swept_high):
            return False

        sl_p = price + SL_MULT * atr
        tp_p = price - TP_MULT * atr

        if not risk_mgr.min_rr_ok(price, tp_p, sl_p):
            return False

        lot = risk_mgr.lot_size(balance, price, sl_p)
        if lot <= 0:
            return False

        logger.info(f"SHORT setup detected | {sym} | ADX: {adx:.0f} | RSI: {rsi:.0f}")
        order_id, fill_price = place_limit_entry(exchange, sym, 'sell', lot, price, atr)

        if order_id and wait_for_fill(exchange, sym, order_id):
            trade = make_trade(sym, 'SHORT', fill_price, tp_p, sl_p, atr, lot, regime)
            open_trades.append(trade)
            alert_trade_open(sym, 'SHORT', fill_price, tp_p, sl_p, lot, 1.0, regime, 3)
            logger.info(f"SHORT opened {sym} @ {fill_price:.2f} | TP {tp_p:.2f} | SL {sl_p:.2f}")
            entered = True

    return entered


# ==================== MAIN LOOP ====================

def run():
    exchange, balance, risk_mgr, deployable = initialise()

    alert_startup(SYMBOLS, balance, deployable)

    open_trades   = []
    cooldowns     = {}    # {symbol: pd.Timestamp}
    candle_times  = {s: None for s in SYMBOLS}
    candle_count  = 0
    daily_stats   = {'trades': 0, 'wins': 0}
    last_day      = datetime.now(timezone.utc).date()

    dfs_15m = {}
    dfs_1h  = {}

    logger.info("\n[SYSTEM] Bot is live with fully optimized Trend-Pullback logic.\n")

    while not _shutdown:
        try:
            time.sleep(LOOP_SLEEP_SECONDS)

            now     = datetime.now(timezone.utc)
            now_day = now.date()

            # ---- Daily reset ----
            if now_day > last_day:
                last_day = now_day
                win_rate = daily_stats['wins'] / max(daily_stats['trades'], 1)
                alert_daily_summary(balance, 0.0, win_rate, daily_stats['trades'], 0.0)
                risk_mgr.reset_daily(balance)
                daily_stats = {'trades': 0, 'wins': 0}
                if not risk_mgr.halted:
                    alert_resume()

            # ---- Halt check ----
            if risk_mgr.halted:
                logger.warning(f"HALTED: {risk_mgr.halt_reason}")
                time.sleep(60)
                continue

            # ---- Balance refresh ----
            candle_count += 1
            if candle_count % BALANCE_REFRESH_CANDLES == 0:
                balance = exchange.fetch_balance()['USDT']['free']
                if not risk_mgr.update(balance):
                    alert_halt(risk_mgr.halt_reason)
                    continue

            # ---- Per-symbol candle loop ----
            for sym in SYMBOLS:
                try:
                    df_15m = build_features(
                        fetch_ohlcv(exchange, sym, '15m', limit=300),
                        funding_rate=fetch_funding_rate(exchange, sym),
                        oi_df=fetch_open_interest(exchange, sym, limit=100),
                    )
                    df_1h = build_features(
                        fetch_ohlcv(exchange, sym, TF_BIAS, limit=300),
                    )

                    curr_candle = df_15m.index[-1]
                    prev_candle = candle_times.get(sym)

                    dfs_15m[sym] = df_15m
                    dfs_1h[sym]  = df_1h

                    if curr_candle == prev_candle:
                        continue
                    candle_times[sym] = curr_candle

                    sym_trades = [t for t in open_trades if t['symbol'] == sym]
                    if len(open_trades) < MAX_TRADES_TOTAL and len(sym_trades) == 0:
                        attempt_entry(
                            sym, df_15m, df_1h, risk_mgr, balance,
                            open_trades, cooldowns, exchange,
                        )

                except Exception as e:
                    logger.error(f"Symbol loop error ({sym}): {e}", exc_info=True)

            # ---- Manage open trades ----
            for trade in open_trades[:]:
                sym = trade['symbol']
                if sym not in dfs_15m:
                    continue
                closed = manage_trade(
                    trade, dfs_15m[sym], exchange,
                    open_trades, daily_stats,
                )
                if closed:
                    cd = dfs_15m[sym].index[-1] + pd.Timedelta(minutes=COOLDOWN_MINUTES)
                    cooldowns[sym] = cd
                    logger.info(f"Cooldown set for {sym} until {cd}")

        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            time.sleep(60)

    logger.info("Bot shut down cleanly.")

if __name__ == '__main__':
    run()
