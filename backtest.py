"""
Standalone backtester. Run directly to evaluate the strategy before going live.

Usage:
    python backtest.py

Produces:
    - Console summary
    - logs/backtest_trades.csv
    - logs/backtest_equity.csv
"""

import os
import csv
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    SYMBOLS, TF_SIGNAL, TF_BIAS,
    RISK_PERCENT, FEE_MAKER, FEE_TAKER,
    MIN_RR, LOG_DIR,
)
from data_feed import connect, fetch_ohlcv_bulk, fetch_open_interest, fetch_funding_rate
from features import build_features, htf_bias, FEAT_COLS
from labels import build_labels, optimize_multipliers
from ict import break_of_structure, liquidity_sweep, fvg, order_block, confluence_score
from validation import trade_stats, sharpe_ratio

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

SLIPPAGE = 0.0005   # 0.05% per side
LOOKAHEAD = 12


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
            if long_s >= 3 and rsi < 65 and bias == 'BULL':
                sl_p = price - self.sl_mult * atr
                tp_p = price + self.tp_mult * atr
                rr   = (tp_p - price) / (price - sl_p + 1e-10)
                if rr >= MIN_RR:
                    entry_p = price * (1 + SLIPPAGE)
                    open_trade = {
                        'type': 'LONG', 'entry': entry_p,
                        'tp': tp_p, 'sl': sl_p,
                        'atr': atr, 'size': 1.0,
                        'open_ts': ts, 'partial': False,
                    }

            # Short setup
            elif short_s >= 3 and rsi > 35 and bias == 'BEAR':
                sl_p = price + self.sl_mult * atr
                tp_p = price - self.tp_mult * atr
                rr   = (price - tp_p) / (sl_p - price + 1e-10)
                if rr >= MIN_RR:
                    entry_p = price * (1 - SLIPPAGE)
                    open_trade = {
                        'type': 'SHORT', 'entry': entry_p,
                        'tp': tp_p, 'sl': sl_p,
                        'atr': atr, 'size': 1.0,
                        'open_ts': ts, 'partial': False,
                    }

        return self.trades, np.array(self.equity)

    def _record(self, trade, exit_price, exit_reason, ts):
        if trade['type'] == 'LONG':
            raw_r = (exit_price - trade['entry']) / (trade['entry'] - trade['sl'] + 1e-10)
        else:
            raw_r = (trade['entry'] - exit_price) / (trade['sl'] - trade['entry'] + 1e-10)

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

    print(f"  Saved: {trade_file}")
    print(f"  Saved: {equity_file}")


def print_summary(symbol, trades, equity):
    if not trades:
        print(f"\n{symbol}: No trades generated.")
        return

    returns = np.array([t['r_multiple'] for t in trades])
    stats   = trade_stats(returns)
    wins    = [t for t in trades if t['r_multiple'] > 0]
    losses  = [t for t in trades if t['r_multiple'] <= 0]
    final   = equity[-1]

    print(f"\n{'='*55}")
    print(f"  {symbol} Backtest Results")
    print(f"{'='*55}")
    print(f"  Total trades     : {stats['trades']}")
    print(f"  Win rate         : {stats['win_rate']*100:.1f}%")
    print(f"  Avg R per trade  : {stats['avg_r']:+.3f}R")
    print(f"  Sharpe (annualised): {stats['sharpe']:+.2f}")
    print(f"  Max drawdown     : {stats['max_dd']*100:.1f}%")
    print(f"  Profit factor    : {final:.3f}x")
    print(f"  Avg win          : {np.mean([t['r_multiple'] for t in wins]):.3f}R" if wins else "  Avg win          : —")
    print(f"  Avg loss         : {np.mean([t['r_multiple'] for t in losses]):.3f}R" if losses else "  Avg loss          : —")
    print(f"{'='*55}")


if __name__ == '__main__':
    exchange = connect()

    for sym in SYMBOLS:
        print(f"\nFetching data for {sym}...")
        df_15m = fetch_ohlcv_bulk(exchange, sym, TF_SIGNAL, total_limit=3000)
        df_1h  = fetch_ohlcv_bulk(exchange, sym, TF_BIAS,   total_limit=1000)

        oi   = fetch_open_interest(exchange, sym)
        fr   = fetch_funding_rate(exchange,  sym)

        df_15m = build_features(df_15m, funding_rate=fr, oi_df=oi)
        df_1h  = build_features(df_1h,  funding_rate=fr)

        # Align 1h to 15m index
        df_1h = df_1h.reindex(df_15m.index, method='ffill').dropna()

        print(f"Optimising multipliers...")
        tp_mult, sl_mult = optimize_multipliers(df_15m)

        bt     = Backtester(sym, df_15m, df_1h, tp_mult, sl_mult)
        trades, equity = bt.run()

        print_summary(sym, trades, equity)
        save_results(sym, trades, equity)
