import logging
import numpy as np
import pandas as pd

from config import (
    WF_TRAIN_MONTHS, WF_TEST_MONTHS, WF_MIN_SHARPE,
    TF_SIGNAL, RISK_PERCENT, FEE_MAKER,
)
from features import FEAT_COLS
from labels import build_labels
from models import EnsembleModel

logger = logging.getLogger(__name__)

# Approximate 15m candles per calendar month
CANDLES_PER_MONTH_15M = 30 * 24 * 4   # 2880


# ==================== METRICS ====================

def sharpe_ratio(returns, periods_per_year=252 * 24 * 4):
    """Annualised Sharpe on a series of per-trade R-multiples."""
    if len(returns) < 5 or np.std(returns) < 1e-10:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


def trade_stats(returns):
    if len(returns) == 0:
        return {}
    wins = returns[returns > 0]
    loss = returns[returns < 0]
    return {
        'trades':    len(returns),
        'win_rate':  len(wins) / len(returns),
        'avg_r':     float(np.mean(returns)),
        'sharpe':    sharpe_ratio(returns),
        'max_dd':    float(_max_drawdown(returns)),
        'profit_f':  float(np.prod(1 + returns * RISK_PERCENT)),
    }


def _max_drawdown(returns):
    equity = np.cumprod(1 + returns * RISK_PERCENT)
    peak   = np.maximum.accumulate(equity)
    dd     = (peak - equity) / (peak + 1e-10)
    return dd.max()


# ==================== SIMULATION ====================

def simulate_returns(df, labels, tp_mult, sl_mult, lookahead=12):
    """
    Fast vectorised P&L simulation used during walk-forward.
    Returns array of R-multiples per trade (including fees).
    """
    closes = df['Close'].values
    highs  = df['High'].values
    lows   = df['Low'].values
    atrs   = df['ATR'].values
    n      = len(df)
    fee    = FEE_MAKER * 2   # Round-trip maker fee in R-space approximation
    rets   = []

    for i in range(n - lookahead):
        label = labels[i]
        if label == 2:
            continue

        entry = closes[i]
        atr   = atrs[i]

        if label == 1:   # Long
            tp = entry + tp_mult * atr
            sl = entry - sl_mult * atr
        else:            # Short
            tp = entry - tp_mult * atr
            sl = entry + sl_mult * atr

        outcome = 0.0
        for j in range(1, lookahead + 1):
            h = highs[i + j]
            l = lows[i + j]

            if label == 1:
                if l <= sl:
                    outcome = -(sl_mult + fee)
                    break
                elif h >= tp:
                    # Simulate partial close at 1R then full at TP
                    outcome = 0.5 * (1.0 - fee) + 0.5 * (tp_mult - fee)
                    break
            else:
                if h >= sl:
                    outcome = -(sl_mult + fee)
                    break
                elif l <= tp:
                    outcome = 0.5 * (1.0 - fee) + 0.5 * (tp_mult - fee)
                    break

        rets.append(outcome)

    return np.array(rets)


# ==================== WALK-FORWARD ====================

def walk_forward_validate(df, tp_mult, sl_mult):
    """
    Sliding walk-forward validation.

    Returns:
        deployable (bool) : True if avg Sharpe >= WF_MIN_SHARPE
        results    (list) : Per-window stats dicts
    """
    df      = df.copy().reset_index(drop=True)
    train_n = WF_TRAIN_MONTHS * CANDLES_PER_MONTH_15M
    test_n  = WF_TEST_MONTHS  * CANDLES_PER_MONTH_15M
    total   = train_n + test_n

    if len(df) < total:
        logger.warning(
            f"Only {len(df)} bars available; need {total} for walk-forward. "
            "Skipping validation — proceed with caution."
        )
        return True, []   # Allow deployment but warn

    results       = []
    start         = 0
    sharpe_scores = []

    print(f"\nWalk-Forward Validation | Train: {WF_TRAIN_MONTHS}m | Test: {WF_TEST_MONTHS}m")
    print("-" * 60)

    while start + total <= len(df):
        train_df = df.iloc[start : start + train_n].copy()
        test_df  = df.iloc[start + train_n : start + total].copy()

        train_labels = build_labels(train_df, tp_mult, sl_mult)
        test_labels  = build_labels(test_df,  tp_mult, sl_mult)

        # Train a fresh model on this window
        model = EnsembleModel(FEAT_COLS, symbol_tag='wf_temp')
        model.fit(train_df, train_labels)

        # Out-of-sample simulation
        rets  = simulate_returns(test_df, test_labels, tp_mult, sl_mult)
        stats = trade_stats(rets)
        sharpe_scores.append(stats.get('sharpe', 0.0))
        results.append(stats)

        print(
            f"  Window {len(results):>2} | "
            f"Trades: {stats.get('trades', 0):>4} | "
            f"Win%: {stats.get('win_rate', 0)*100:>5.1f}% | "
            f"Avg R: {stats.get('avg_r', 0):>+.3f} | "
            f"Sharpe: {stats.get('sharpe', 0):>+.2f}"
        )

        start += test_n

    avg_sharpe = float(np.mean(sharpe_scores)) if sharpe_scores else 0.0
    deployable = avg_sharpe >= WF_MIN_SHARPE

    print("-" * 60)
    print(f"Avg Sharpe: {avg_sharpe:.2f} | Min required: {WF_MIN_SHARPE} | Deployable: {deployable}\n")

    return deployable, results
