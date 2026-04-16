import pandas as pd
import numpy as np


# ==================== SWING POINTS ====================

def swing_points(df, lookback=2):
    highs = df['High']
    lows  = df['Low']

    mask_high = (highs.shift(lookback) < highs.shift(1)) & (highs.shift(0) < highs.shift(1))
    swing_high = highs.shift(1)[mask_high].dropna()

    mask_low = (lows.shift(lookback) > lows.shift(1)) & (lows.shift(0) > lows.shift(1))
    swing_low = lows.shift(1)[mask_low].dropna()

    return swing_high, swing_low


# ==================== BREAK OF STRUCTURE ====================

def break_of_structure(df):
    """
    BOS_UP:   last close broke above the most recent confirmed swing high → bullish
    BOS_DOWN: last close broke below the most recent confirmed swing low  → bearish
    """
    swing_high, swing_low = swing_points(df)

    if swing_high.empty or swing_low.empty:
        return False, False

    last_close = df['Close'].iloc[-1]
    bos_up     = last_close > swing_high.iloc[-1]
    bos_down   = last_close < swing_low.iloc[-1]
    return bos_up, bos_down


# ==================== LIQUIDITY SWEEP ====================

def liquidity_sweep(df, period=20):
    """
    Sweep HIGH: current candle's high exceeded the prior 20-bar high
                then closed back inside → engineered liquidity grab above.
    Sweep LOW:  current candle's low undercut the prior 20-bar low
                then closed back inside → engineered liquidity grab below.
    We use iloc[-2] for the rolling max/min to avoid the current candle
    inflating its own reference level.
    """
    rolling_high = df['High'].rolling(period).max().iloc[-2]
    rolling_low  = df['Low'].rolling(period).min().iloc[-2]

    curr_high = df['High'].iloc[-1]
    curr_low  = df['Low'].iloc[-1]
    curr_close = df['Close'].iloc[-1]

    swept_high = curr_high > rolling_high and curr_close < rolling_high
    swept_low  = curr_low  < rolling_low  and curr_close > rolling_low

    return swept_high, swept_low


# ==================== FAIR VALUE GAP ====================

def fvg(df):
    """
    Bullish FVG: candle[-1].Low > candle[-3].High  (gap up — price left empty air below)
    Bearish FVG: candle[-1].High < candle[-3].Low  (gap down — price left empty air above)
    FVG signals an imbalance that price often returns to before continuing.
    """
    gap_up   = df['Low'].iloc[-1]  > df['High'].iloc[-3]
    gap_down = df['High'].iloc[-1] < df['Low'].iloc[-3]
    return gap_up, gap_down


# ==================== ORDER BLOCK ====================

def order_block(df):
    """
    Bullish OB: last bearish candle before a strong up move.
    Bearish OB: last bullish candle before a strong down move.
    Simple proxy: look for engulfing move in the last 5 candles.
    Returns (bull_ob_price, bear_ob_price) or (None, None).
    """
    n = len(df)
    if n < 6:
        return None, None

    bull_ob = None
    bear_ob = None

    for i in range(n - 5, n - 1):
        body = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
        atr  = df['ATR'].iloc[i]

        # Strong move = body > 1.5x ATR
        if body > 1.5 * atr:
            if df['Close'].iloc[i] > df['Open'].iloc[i]:
                # Bullish engulf — OB is the last bearish candle before it
                if i > 0 and df['Close'].iloc[i-1] < df['Open'].iloc[i-1]:
                    bull_ob = df['Low'].iloc[i-1]
            else:
                # Bearish engulf — OB is the last bullish candle before it
                if i > 0 and df['Close'].iloc[i-1] > df['Open'].iloc[i-1]:
                    bear_ob = df['High'].iloc[i-1]

    return bull_ob, bear_ob


# ==================== CONFLUENCE SCORE ====================

def confluence_score(bias, bos_up, bos_down, swept_high, swept_low,
                     gap_up, gap_down, bull_ob, bear_ob, price):
    """
    Returns (long_score, short_score) as integers 0–4.
    Higher = more ICT confluence.
    NOTE: Bias is NOT counted here (it's already checked in entry logic).
    Signals: BOS, Liquidity Sweep, FVG, Order Block.
    """
    long_score  = 0
    short_score = 0

    if bos_up:
        long_score  += 1
    if bos_down:
        short_score += 1

    if swept_low:                 # Liquidity taken below → reversal up
        long_score  += 1
    if swept_high:                # Liquidity taken above → reversal down
        short_score += 1

    if gap_up:
        long_score  += 1
    if gap_down:
        short_score += 1

    if bull_ob and price <= bull_ob * 1.005:   # Price inside / near bull OB
        long_score  += 1
    if bear_ob and price >= bear_ob * 0.995:   # Price inside / near bear OB
        short_score += 1

    return long_score, short_score
