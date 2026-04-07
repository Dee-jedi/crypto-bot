"""
Main live trading bot.

Run:
    python bot.py

Startup sequence:
    1. Connect to Binance Futures (testnet if TESTNET=True in config)
    2. Fetch historical data for all symbols
    3. Optimise TP/SL multipliers
    4. Train or load models
    5. Run walk-forward validation
    6. Enter main loop

Per-candle loop:
    - Refresh features
    - Check risk manager (halt conditions)
    - Per symbol: detect regime, score ICT confluence, get ML signal
    - Enter via limit order if all filters pass
    - Manage open trades (partial close, trail, TP/SL)
    - Online-learn after each closed trade
    - Daily reset at UTC midnight
    - Telegram alerts throughout
"""

import os
import sys
import time
import logging
import signal
import threading
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque

import pandas as pd
import numpy as np

from config import (
    SYMBOLS, TF_SIGNAL, TF_BIAS, SEQ_LEN,
    MAX_TRADES_TOTAL, COOLDOWN_MINUTES,
    CONFIDENCE_THRESH, CONFIDENCE_FLOOR,
    BALANCE_REFRESH_CANDLES, LOOP_SLEEP_SECONDS,
    LOG_DIR, SESSION_START_UTC, SESSION_END_UTC,
)
from data_feed import (
    connect, fetch_ohlcv, fetch_ohlcv_bulk,
    fetch_funding_rate, fetch_open_interest,
)
from features import build_features, htf_bias, FEAT_COLS
from labels import build_labels, optimize_multipliers
from ict import break_of_structure, liquidity_sweep, fvg, order_block, confluence_score
from models import EnsembleModel
from validation import walk_forward_validate
from risk import RiskManager, correlated_exposure, log_trade
from execution import place_limit_entry, wait_for_fill, close_partial, close_full
from alerts import (
    alert_trade_open, alert_trade_close, alert_partial_close,
    alert_halt, alert_resume, alert_confidence_drift,
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


# ==================== ASYNC HELPER ====================

def _async_model_update(model, x_tensor, result):
    """Runs online learning backprop and save in a background thread."""
    try:
        model.online_update(x_tensor, result)
        model.record_outcome(result)
        model.save()
    except Exception as e:
        logger.error(f"Async model update failed: {e}", exc_info=True)


# ==================== TRADE FACTORY ====================

def make_trade(symbol, trade_type, entry, tp, sl, atr, lot_size, x_tensor, conf, regime):
    return {
        'symbol':    symbol,
        'type':      trade_type,
        'entry':     entry,
        'tp':        tp,
        'sl':        sl,
        'initial_sl': sl,
        'atr':       atr,
        'lot_size':  lot_size,
        'remaining': lot_size,   # Reduces after partial close
        'partial':   False,
        'x_tensor':  x_tensor,
        'confidence': conf,
        'regime':    regime,
        'open_time': datetime.now(timezone.utc),
    }


# ==================== CONFIDENCE TRACKER ====================

class ConfidenceTracker:
    def __init__(self, window=20):
        self._history = defaultdict(lambda: deque(maxlen=window))

    def record(self, symbol, conf):
        self._history[symbol].append(conf)

    def avg(self, symbol):
        h = self._history[symbol]
        return float(np.mean(h)) if h else 1.0

    def is_drifting(self, symbol):
        return self.avg(symbol) < CONFIDENCE_FLOOR


# ==================== INITIALISE ====================

def initialise():
    logger.info("Connecting to exchange...")
    exchange = connect()

    balance = exchange.fetch_balance()['USDT']['free']
    logger.info(f"Connected | Balance: ${balance:,.2f}")

    risk_mgr  = RiskManager(balance)
    models    = {}
    tp_sl     = {}
    conf_tracker = ConfidenceTracker()

    for sym in SYMBOLS:
        tag = sym.replace('/', '')
        logger.info(f"Fetching historical data: {sym}...")

        df_15m = fetch_ohlcv_bulk(exchange, sym, TF_SIGNAL, total_limit=3000)
        oi     = fetch_open_interest(exchange, sym)
        fr     = fetch_funding_rate(exchange, sym)
        df_15m = build_features(df_15m, funding_rate=fr, oi_df=oi)

        logger.info(f"Optimising multipliers: {sym}...")
        tp, sl      = optimize_multipliers(df_15m)
        tp_sl[sym]  = (tp, sl)

        model = EnsembleModel(FEAT_COLS, symbol_tag=tag)

        if model.load():
            logger.info(f"Loaded saved model: {sym}")
        else:
            logger.info(f"Training new model: {sym}...")
            labels = build_labels(df_15m, tp, sl)
            model.fit(df_15m, labels)
            model.save()
            logger.info(f"Model trained and saved: {sym}")

        # Walk-forward validation
        logger.info(f"Running walk-forward validation: {sym}...")
        deployable, wf_results = walk_forward_validate(df_15m, tp, sl)
        if not deployable:
            logger.warning(f"Walk-forward FAILED for {sym}. Deploying with caution.")

        models[sym] = model

    return exchange, balance, risk_mgr, models, tp_sl, conf_tracker, deployable


# ==================== TRADE MANAGEMENT ====================

def manage_trade(trade, df_15m, exchange, models, open_trades, daily_stats):
    """
    Check breakeven / trail / TP / SL for a single open trade.
    Modifies trade dict in-place.
    Returns True if the trade was closed, False if still open.
    """
    sym   = trade['symbol']
    h     = df_15m['High'].iloc[-1]
    l     = df_15m['Low'].iloc[-1]
    price = df_15m['Close'].iloc[-1]
    atr   = trade['atr']
    t     = trade

    # ---- Partial close at 1R ----
    if not t['partial']:
        reached_1r = (
            (t['type'] == 'LONG'  and h >= t['entry'] + atr) or
            (t['type'] == 'SHORT' and l <= t['entry'] - atr)
        )
        if reached_1r:
            closed_qty = close_partial(exchange, sym, t['type'], t['remaining'])
            if closed_qty > 0:
                t['partial']   = True
                t['remaining'] -= closed_qty
                t['sl']        = t['entry']   # Move to breakeven
                r_so_far       = 1.0
                alert_partial_close(sym, t['type'], price, r_so_far)
                logger.info(f"Partial close {sym} {t['type']} | SL → breakeven")

    # ---- Trail stop ----
    if t['type'] == 'LONG' and h >= t['entry'] + 2 * atr:
        t['sl'] = max(t['sl'], h - atr)
    if t['type'] == 'SHORT' and l <= t['entry'] - 2 * atr:
        t['sl'] = min(t['sl'], l + atr)

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

            threading.Thread(target=_async_model_update, args=(models[sym], t['x_tensor'], 0)).start()

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

            threading.Thread(target=_async_model_update, args=(models[sym], t['x_tensor'], 1)).start()

            log_trade(sym, t['type'], t['entry'], exit_price, 1, t['lot_size'], r_raw)
            alert_trade_close(sym, t['type'], t['entry'], exit_price, 1, r_raw)
            daily_stats['trades'] += 1
            daily_stats['wins']   += 1
            open_trades.remove(t)
            logger.info(f"TP closed {sym} {t['type']} @ {exit_price:.2f} | {r_raw:+.2f}R")
            return True

    return False


# ==================== ENTRY LOGIC ====================

def attempt_entry(sym, df_15m, df_1h, model, tp_sl, risk_mgr,
                  balance, open_trades, cooldowns, exchange):
    """
    Evaluates entry conditions for one symbol.
    Returns True if a trade was initiated.
    """
    tp_mult, sl_mult = tp_sl[sym]
    price            = df_15m['Close'].iloc[-1]
    atr              = df_15m['ATR'].iloc[-1]
    rsi              = df_15m['RSI'].iloc[-1]
    ts               = df_15m.index[-1]
    hour_utc         = ts.hour

    # ---- Time filter: London/NY overlap only ----
    if not (SESSION_START_UTC <= hour_utc < SESSION_END_UTC):
        return False

    # ---- Cooldown ----
    cd = cooldowns.get(sym)
    if cd and ts < cd:
        return False

    # ---- HTF bias ----
    bias = htf_bias(df_1h)

    # ---- ICT signals ----
    window              = df_15m.iloc[-80:]
    bos_up,  bos_down  = break_of_structure(window)
    sw_high, sw_low    = liquidity_sweep(window)
    gap_up,  gap_down  = fvg(window)
    bull_ob, bear_ob   = order_block(window)
    long_s, short_s    = confluence_score(
        bias, bos_up, bos_down, sw_high, sw_low,
        gap_up, gap_down, bull_ob, bear_ob, price,
    )

    # ---- ML signal ----
    pred, conf, x_tensor = model.predict(df_15m)
    regime               = int(df_15m['Regime'].iloc[-1])

    # Confidence gate
    if conf < CONFIDENCE_THRESH:
        return False

    # ---- Correlation guard ----
    if correlated_exposure(open_trades):
        return False

    entered = False

    # ---- LONG ----
    if (
        pred == 1
        and long_s >= 3
        and bias == 'BULL'
        and rsi < 65
        and probs_ok(conf, 'long')
    ):
        sl_p = price - sl_mult * atr
        tp_p = price + tp_mult * atr

        if not risk_mgr.min_rr_ok(price, tp_p, sl_p):
            logger.info(f"LONG skipped — RR below minimum ({sym})")
            return False

        lot = risk_mgr.lot_size(balance, price, sl_p)
        if lot <= 0:
            return False

        logger.info(f"LONG setup detected | {sym} | ICT: {long_s}/5 | Conf: {conf:.0%}")
        order_id, fill_price = place_limit_entry(exchange, sym, 'buy', lot, price, atr)

        if order_id and wait_for_fill(exchange, sym, order_id):
            trade = make_trade(sym, 'LONG', fill_price, tp_p, sl_p, atr, lot, x_tensor, conf, regime)
            open_trades.append(trade)
            alert_trade_open(sym, 'LONG', fill_price, tp_p, sl_p, lot, conf, regime, long_s)
            logger.info(f"LONG opened {sym} @ {fill_price:.2f} | TP {tp_p:.2f} | SL {sl_p:.2f}")
            entered = True
        else:
            logger.info(f"LONG limit not filled ({sym})")

    # ---- SHORT ----
    elif (
        pred == 0
        and short_s >= 3
        and bias == 'BEAR'
        and rsi > 35
        and probs_ok(conf, 'short')
    ):
        sl_p = price + sl_mult * atr
        tp_p = price - tp_mult * atr

        if not risk_mgr.min_rr_ok(price, tp_p, sl_p):
            logger.info(f"SHORT skipped — RR below minimum ({sym})")
            return False

        lot = risk_mgr.lot_size(balance, price, sl_p)
        if lot <= 0:
            return False

        logger.info(f"SHORT setup detected | {sym} | ICT: {short_s}/5 | Conf: {conf:.0%}")
        order_id, fill_price = place_limit_entry(exchange, sym, 'sell', lot, price, atr)

        if order_id and wait_for_fill(exchange, sym, order_id):
            trade = make_trade(sym, 'SHORT', fill_price, tp_p, sl_p, atr, lot, x_tensor, conf, regime)
            open_trades.append(trade)
            alert_trade_open(sym, 'SHORT', fill_price, tp_p, sl_p, lot, conf, regime, short_s)
            logger.info(f"SHORT opened {sym} @ {fill_price:.2f} | TP {tp_p:.2f} | SL {sl_p:.2f}")
            entered = True

    return entered


def probs_ok(conf, direction):
    """Extra confidence gate — can extend with direction-specific logic."""
    return conf >= CONFIDENCE_THRESH


# ==================== MAIN LOOP ====================

def run():
    (exchange, balance, risk_mgr, models,
     tp_sl, conf_tracker, deployable) = initialise()

    alert_startup(SYMBOLS, balance, deployable)

    open_trades   = []
    cooldowns     = {}    # {symbol: pd.Timestamp}
    candle_times  = {s: None for s in SYMBOLS}
    candle_count  = 0
    daily_stats   = {'trades': 0, 'wins': 0}
    last_day      = datetime.now(timezone.utc).date()

    # Cache of latest DataFrames per symbol
    dfs_15m = {}
    dfs_1h  = {}

    logger.info("\n🚀 Bot is live. Waiting for candles...\n")

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

            # ---- Balance refresh (rate-limit-friendly) ----
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

                    # Only act on a new candle close
                    if curr_candle == prev_candle:
                        continue
                    candle_times[sym] = curr_candle

                    # Track model confidence
                    _, conf, _ = models[sym].predict(df_15m)
                    conf_tracker.record(sym, conf)
                    if conf_tracker.is_drifting(sym):
                        alert_confidence_drift(sym, conf_tracker.avg(sym))

                    # Entry (if capacity allows)
                    sym_trades = [t for t in open_trades if t['symbol'] == sym]
                    if (
                        len(open_trades) < MAX_TRADES_TOTAL
                        and len(sym_trades) == 0
                    ):
                        attempt_entry(
                            sym, df_15m, df_1h, models[sym],
                            tp_sl, risk_mgr, balance,
                            open_trades, cooldowns, exchange,
                        )

                except Exception as e:
                    logger.error(f"Symbol loop error ({sym}): {e}", exc_info=True)

            # ---- Manage all open trades ----
            for trade in open_trades[:]:
                sym = trade['symbol']
                if sym not in dfs_15m:
                    continue
                closed = manage_trade(
                    trade, dfs_15m[sym], exchange,
                    models, open_trades, daily_stats,
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
