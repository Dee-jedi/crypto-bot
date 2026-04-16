import pandas as pd
import os
import glob

LOG_DIR = 'logs'

def analyze_per_symbol():
    files = glob.glob(os.path.join(LOG_DIR, 'backtest_trades_*.csv'))
    results = []
    
    for f in files:
        symbol = os.path.basename(f).replace('backtest_trades_', '').replace('.csv', '')
        df = pd.read_csv(f)
        
        if df.empty:
            continue
            
        wins = df[df['r_multiple'] > 0]
        losses = df[df['r_multiple'] <= 0]
        
        win_rate = len(wins) / len(df)
        total_r = df['r_multiple'].sum()
        avg_r = df['r_multiple'].mean()
        
        gross_profit = wins['r_multiple'].sum()
        gross_loss = abs(losses['r_multiple'].sum())
        profit_factor = gross_profit / (gross_loss + 1e-10)
        
        results.append({
            'symbol': symbol,
            'trades': len(df),
            'win_rate': round(win_rate * 100, 1),
            'total_r': round(total_r, 2),
            'avg_r': round(avg_r, 3),
            'profit_factor': round(profit_factor, 2)
        })
        
    res_df = pd.DataFrame(results).sort_values('total_r', ascending=False)
    print(res_df.to_string(index=False))

if __name__ == '__main__':
    analyze_per_symbol()
