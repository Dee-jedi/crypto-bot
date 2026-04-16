import pandas as pd
import numpy as np
import os

# Paths
TRADES_FILE = r'logs/backtest_trades_BTCUSDT.csv'
DATA_FILE   = r'data/cache/BTCUSDT_15m_60m.csv'

def analyze_deep():
    if not os.path.exists(TRADES_FILE) or not os.path.exists(DATA_FILE):
        print("Missing trades or data files.")
        return

    # Load trades and data
    trades = pd.read_csv(TRADES_FILE, parse_dates=['open_ts', 'close_ts'])
    data = pd.read_csv(DATA_FILE, index_col='time', parse_dates=True)
    
    # We need to compute ATR to check old SL/TP distances
    # Copying basic ATR logic from features.py
    def compute_atr(df, period=14):
        prev_close = df['Close'].shift(1)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - prev_close).abs(),
            (df['Low']  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    data['ATR'] = compute_atr(data)

    print(f"Total trades to analyze: {len(trades)}")
    
    results = []

    for idx, trade in trades.iterrows():
        # Get data slice from open to close
        # Add some buffer after close to see if TP would have hit
        start_ts = trade['open_ts']
        # Look ahead up to 96 bars (24h) from entry as in the backtester
        end_ts = start_ts + pd.Timedelta(hours=24)
        
        trade_slice = data.loc[start_ts:end_ts]
        if trade_slice.empty:
            continue
            
        entry_p = trade['entry']
        atr = data.loc[start_ts, 'ATR']
        
        # Original targets
        if trade['type'] == 'LONG':
            target_tp = entry_p + (2.0 * atr)
            target_sl = entry_p - (1.0 * atr)
            target_be = entry_p + (1.0 * atr)
        else:
            target_tp = entry_p - (2.0 * atr)
            target_sl = entry_p + (1.0 * atr)
            target_be = entry_p - (1.0 * atr)
            
        # Analyze what happened in the next 24 hours
        hit_tp = False
        hit_sl = False
        hit_be = False
        tp_time = None
        sl_time = None
        be_time = None
        
        for t, row in trade_slice.iterrows():
            if trade['type'] == 'LONG':
                if not hit_be and row['High'] >= target_be:
                    hit_be = True
                    be_time = t
                if not hit_tp and row['High'] >= target_tp:
                    hit_tp = True
                    tp_time = t
                if not hit_sl and row['Low'] <= target_sl:
                    hit_sl = True
                    sl_time = t
            else: # SHORT
                if not hit_be and row['Low'] <= target_be:
                    hit_be = True
                    be_time = t
                if not hit_tp and row['Low'] <= target_tp:
                    hit_tp = True
                    tp_time = t
                if not hit_sl and row['High'] >= target_sl:
                    hit_sl = True
                    sl_time = t
                    
            # Order matters: SL first or TP first?
            # In a real backtest, we check if SL hits before TP in a single bar
            # Here we just want to see if TP *ever* hits before SL
            if hit_sl or hit_tp:
                break
        
        # Outcome classification
        outcome = "UNKNOWN"
        if hit_tp and (not hit_sl or tp_time <= sl_time):
            outcome = "WIN"
        elif hit_sl and (not hit_tp or sl_time < tp_time):
            outcome = "LOSS"
        else:
            outcome = "TIMEOUT"
            
        # Analysis of the "Breakeven" trap
        # Did it hit BE, then hit SL(BE), then eventually hit TP without hitting original SL?
        be_reached_tp = False
        if hit_be:
             # Look further in the slice
             for t, row in trade_slice.loc[be_time:].iterrows():
                 if trade['type'] == 'LONG':
                     if row['Low'] <= entry_p: # Hit BE
                         # Now check if it hits TP before original SL
                         for t2, row2 in trade_slice.loc[t:].iterrows():
                             if row2['High'] >= target_tp:
                                 be_reached_tp = True
                                 break
                             if row2['Low'] <= target_sl:
                                 break
                         break
                 else:
                     if row['High'] >= entry_p:
                         for t2, row2 in trade_slice.loc[t:].iterrows():
                             if row2['Low'] <= target_tp:
                                 be_reached_tp = True
                                 break
                             if row2['High'] >= target_sl:
                                 break
                         break

        results.append({
            'outcome': outcome,
            'be_reached_tp': be_reached_tp,
            'hit_be': hit_be,
            'type': trade['type']
        })

    results_df = pd.DataFrame(results)
    print("\n--- Deep Analysis Summary ---")
    print(f"Wins:    {len(results_df[results_df['outcome'] == 'WIN'])}")
    print(f"Losses:  {len(results_df[results_df['outcome'] == 'LOSS'])}")
    print(f"Breakeven Trap Count: {results_df['be_reached_tp'].sum()}")
    
    # Calculate win rate if no BE logic was used
    potential_wins = len(results_df[results_df['outcome'] == 'WIN'])
    potential_losses = len(results_df[results_df['outcome'] == 'LOSS'])
    total = potential_wins + potential_losses
    if total > 0:
        print(f"Potential Win Rate (No BE): {potential_wins / total * 100:.1f}%")
        print(f"Win Rate including BE Recovery: {(potential_wins + results_df['be_reached_tp'].sum()) / total * 100:.1f}%")

if __name__ == '__main__':
    analyze_deep()
