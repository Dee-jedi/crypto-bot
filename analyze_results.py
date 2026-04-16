import pandas as pd
import numpy as np
import os

LOG_DIR = 'logs'

def analyze():
    combined_trades = []
    
    # Load all trade files
    for file in os.listdir(LOG_DIR):
        if file.startswith('backtest_trades_') and file.endswith('.csv'):
            path = os.path.join(LOG_DIR, file)
            df = pd.read_csv(path)
            combined_trades.append(df)
            
    if not combined_trades:
        print("No trades found.")
        return
        
    all_trades = pd.concat(combined_trades)
    all_trades = all_trades.sort_values('close_ts') # Sort by time to see portfolio curve
    
    total_trades = len(all_trades)
    win_rate = (all_trades['r_multiple'] > 0).mean()
    avg_r = all_trades['r_multiple'].mean()
    
    # Portfolio equity calculation
    # Starting with 10k, risking 1% per trade
    balance = 10000.0
    equity = [balance]
    max_equity = balance
    max_dd = 0
    
    for r in all_trades['r_multiple']:
        balance *= (1 + r * 0.02)
        equity.append(balance)
        max_equity = max(max_equity, balance)
        dd = (max_equity - balance) / max_equity
        max_dd = max(max_dd, dd)
        
    print(f"--- Portfolio Summary (BTC+ETH) ---")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate:     {win_rate*100:.1f}%")
    print(f"Avg R:        {avg_r:+.3f}R")
    print(f"Final Balance: ${balance:.2f}")
    print(f"Total Return:  {(balance-10000)/100:.2f}%")
    print(f"Max Drawdown:  {max_dd*100:.1f}%")
    print(f"Sharpe (approx): {avg_r / (all_trades['r_multiple'].std() + 1e-10) * np.sqrt(total_trades):.2f}")

if __name__ == '__main__':
    analyze()
