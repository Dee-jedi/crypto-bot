import numpy as np


def build_labels(df, tp_mult=2.0, sl_mult=1.0, lookahead=12):
    """
    Labels each bar as:
        1  = long opportunity  (long TP hit before long SL, short direction fails)
        0  = short opportunity (short TP hit before short SL, long direction fails)
        2  = neutral           (ambiguous, conflicting, or neither triggers)

    Both long and short are evaluated independently per bar.
    A label is only assigned when one direction clearly wins.
    The 12-bar tail is padded with 2 (neutral) — no synthetic signal injected.
    """
    closes = df['Close'].values
    highs  = df['High'].values
    lows   = df['Low'].values
    atrs   = df['ATR'].values
    n      = len(df)
    labels = []

    for i in range(n - lookahead):
        entry = closes[i]
        atr   = atrs[i]

        long_tp  = entry + tp_mult * atr
        long_sl  = entry - sl_mult * atr
        short_tp = entry - tp_mult * atr
        short_sl = entry + sl_mult * atr

        long_result  = None   # 'win' | 'fail'
        short_result = None

        for j in range(1, lookahead + 1):
            h = highs[i + j]
            l = lows[i + j]

            if long_result is None:
                if l <= long_sl:
                    long_result = 'fail'
                elif h >= long_tp:
                    long_result = 'win'

            if short_result is None:
                if h >= short_sl:
                    short_result = 'fail'
                elif l <= short_tp:
                    short_result = 'win'

            if long_result and short_result:
                break

        if long_result == 'win' and short_result != 'win':
            labels.append(1)
        elif short_result == 'win' and long_result != 'win':
            labels.append(0)
        else:
            labels.append(2)   # Conflicting or neither hit

    # Tail padding — genuinely unknown, not a signal
    labels += [2] * lookahead
    return np.array(labels)


def optimize_multipliers(df):
    """
    Grid search over TP/SL multipliers evaluating both long AND short
    directions independently on the same historical data.
    Returns the (tp_mult, sl_mult) pair with the highest total R-multiple.
    """
    best        = (2.0, 1.0)
    best_score  = -999.0

    print("Optimising TP/SL multipliers (bidirectional)...")

    for tp in [1.5, 2.0, 2.5, 3.0]:
        for sl in [0.5, 1.0, 1.5]:
            total = 0.0
            n     = len(df) - 12

            for i in range(n):
                entry = df['Close'].iloc[i]
                atr   = df['ATR'].iloc[i]

                long_tp  = entry + tp * atr
                long_sl  = entry - sl * atr
                short_tp = entry - tp * atr
                short_sl = entry + sl * atr

                long_r  = 0.0
                short_r = 0.0

                for j in range(1, 13):
                    h = df['High'].iloc[i + j]
                    l = df['Low'].iloc[i + j]

                    if long_r == 0.0:
                        if l <= long_sl:
                            long_r = -sl
                        elif h >= long_tp:
                            long_r = tp

                    if short_r == 0.0:
                        if h >= short_sl:
                            short_r = -sl
                        elif l <= short_tp:
                            short_r = tp

                    if long_r != 0.0 and short_r != 0.0:
                        break

                total += long_r + short_r

            if total > best_score:
                best_score = total
                best       = (tp, sl)

    print(f"  Best → TP: {best[0]}x ATR | SL: {best[1]}x ATR | Score: {best_score:.1f}R")
    return best
