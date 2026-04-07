import os
import csv
import logging
import numpy as np
from datetime import datetime, timezone

from config import (
    RISK_PERCENT, MIN_RR,
    DAILY_LOSS_LIMIT, DD_CIRCUIT_BREAK,
    LOG_DIR, EQUITY_LOG,
)

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Tracks equity curve and enforces:
      - Per-trade risk sizing
      - Minimum R:R filter
      - Daily loss limit  (resets at UTC midnight)
      - Drawdown circuit breaker (requires manual/retrain reset)
    """

    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.peak_balance    = initial_balance
        self.daily_start_bal = initial_balance
        self.halted          = False
        self.halt_reason     = ""
        os.makedirs(LOG_DIR, exist_ok=True)

    # ---- balance update ----

    def update(self, current_balance: float) -> bool:
        """
        Call on every balance refresh.
        Returns False (and sets self.halted) if a circuit breaker trips.
        """
        self.peak_balance = max(self.peak_balance, current_balance)

        drawdown   = (self.peak_balance - current_balance) / (self.peak_balance + 1e-10)
        daily_loss = (self.daily_start_bal - current_balance) / (self.daily_start_bal + 1e-10)

        self._log_equity(current_balance, drawdown, daily_loss)

        if drawdown >= DD_CIRCUIT_BREAK:
            reason = (
                f"Drawdown circuit breaker tripped: {drawdown*100:.1f}% "
                f"(limit {DD_CIRCUIT_BREAK*100:.0f}%). "
                "Manual restart required after model retrain."
            )
            self._halt(reason)
            return False

        if daily_loss >= DAILY_LOSS_LIMIT and not self.halted:
            reason = (
                f"Daily loss limit hit: {daily_loss*100:.1f}% "
                f"(limit {DAILY_LOSS_LIMIT*100:.0f}%). "
                "Will resume on next UTC day."
            )
            self._halt(reason)
            return False

        return True

    def reset_daily(self, current_balance: float):
        """Call once at UTC midnight."""
        self.daily_start_bal = current_balance
        # Only lift halt if it was a daily-loss halt, not a drawdown halt
        if self.halted and 'Daily loss' in self.halt_reason:
            self.halted      = False
            self.halt_reason = ""
            logger.info("Daily reset — trading resumed.")

    def _halt(self, reason: str):
        self.halted      = True
        self.halt_reason = reason
        logger.critical(f"TRADING HALTED: {reason}")

    # ---- position sizing ----

    def lot_size(self, balance: float, entry: float, sl_price: float) -> float:
        """
        Risk RISK_PERCENT of balance between entry and stop-loss.
        Returns lot size floored to 3 decimal places.
        """
        risk_amount    = balance * RISK_PERCENT
        price_distance = abs(entry - sl_price)
        if price_distance < 1e-10:
            return 0.0
        size = risk_amount / price_distance
        return float(np.floor(size * 1000) / 1000)

    # ---- filters ----

    def min_rr_ok(self, entry: float, tp: float, sl: float) -> bool:
        """Skip trades that don't offer at least MIN_RR reward:risk."""
        reward = abs(tp - entry)
        risk   = abs(sl - entry)
        if risk < 1e-10:
            return False
        return (reward / risk) >= MIN_RR

    # ---- logging ----

    def _log_equity(self, balance, drawdown, daily_loss):
        file_exists = os.path.isfile(EQUITY_LOG)
        with open(EQUITY_LOG, 'a', newline='') as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(['Timestamp', 'Balance', 'Drawdown_pct', 'Daily_PnL_pct'])
            w.writerow([
                datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                f'{balance:.2f}',
                f'{drawdown*100:.2f}',
                f'{daily_loss*100:.2f}',
            ])


# ==================== CORRELATION FILTER ====================

def correlated_exposure(open_trades: list) -> bool:
    """
    Returns True if adding another trade would pile up correlated directional exposure.
    Now relaxed: It allows simultaneous LONGs or SHORTs on different pairs,
    provided the overall MAX_TRADES_TOTAL limit is respected in the main loop.
    """
    return False


# ==================== PERFORMANCE LOG ====================

def log_trade(symbol, trade_type, entry, exit_price, result, lot_size, r_multiple):
    """Append completed trade to CSV performance log."""
    from config import PERF_LOG
    os.makedirs(LOG_DIR, exist_ok=True)
    file_exists = os.path.isfile(PERF_LOG)
    with open(PERF_LOG, 'a', newline='') as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                'Timestamp', 'Symbol', 'Type', 'Entry', 'Exit',
                'Result', 'Lot', 'R_Multiple',
            ])
        w.writerow([
            datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            symbol, trade_type,
            f'{entry:.2f}', f'{exit_price:.2f}',
            'WIN' if result == 1 else 'LOSS',
            lot_size, f'{r_multiple:.2f}',
        ])
