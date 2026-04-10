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


def optimize_multipliers(df, lookahead=24, min_rr=1.5):
    """
    Grid search over TP/SL multipliers evaluating both long AND short
    directions independently on the same historical data.
    Skips TP/SL combos that don't meet the min R:R ratio.

    Scoring prioritizes WIN RATE over raw R-multiple:
      score = win_rate * avg_win_r  -  loss_rate * avg_loss_r

    This avoids choosing combos with 10% win rate and huge wins
    that are unrealistic in live trading.

    Returns the (tp_mult, sl_mult) pair with the highest score.
    """
    best        = (2.0, 1.5)
    best_score  = -999.0

    print(f"Optimising TP/SL multipliers (lookahead={lookahead}, min_rr={min_rr})...")

    # Wider SL range: research shows 2.0-2.5x ATR is the sweet spot for 15m crypto
    for tp in [1.5, 2.0, 2.5, 3.0]:
        for sl in [1.0, 1.5, 2.0, 2.5]:
            # Skip combos that can't pass MIN_RR
            if tp / sl < min_rr:
                continue

            wins   = 0
            losses = 0
            win_r  = 0.0
            loss_r = 0.0
            n      = len(df) - lookahead

            for i in range(n):
                entry = df['Close'].iloc[i]
                atr   = df['ATR'].iloc[i]

                long_tp  = entry + tp * atr
                long_sl  = entry - sl * atr
                short_tp = entry - tp * atr
                short_sl = entry + sl * atr

                long_r  = 0.0
                short_r = 0.0

                for j in range(1, lookahead + 1):
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

                # Count wins and losses for both directions
                for r in [long_r, short_r]:
                    if r > 0:
                        wins += 1
                        win_r += r
                    elif r < 0:
                        losses += 1
                        loss_r += abs(r)

            total_trades = wins + losses
            if total_trades == 0:
                continue

            win_rate = wins / total_trades
            avg_win  = win_r / max(wins, 1)
            avg_loss = loss_r / max(losses, 1)

            # Score that rewards win rate: expectancy per trade
            # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            score = expectancy * total_trades  # Scale by trade count

            if score > best_score:
                best_score = score
                best       = (tp, sl)
                print(f"    TP:{tp} SL:{sl} | WR:{win_rate:.1%} | "
                      f"AvgW:{avg_win:.2f}R AvgL:{avg_loss:.2f}R | "
                      f"Expectancy:{expectancy:.3f}R | Score:{score:.1f}")

    print(f"  Best -> TP: {best[0]}x ATR | SL: {best[1]}x ATR | Score: {best_score:.1f}")
    return best
