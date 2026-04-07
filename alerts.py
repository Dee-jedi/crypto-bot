import logging
import requests
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


def _send(text: str):
    """Fire-and-forget Telegram message. Silently swallows errors."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(
            url,
            data={'chat_id': TELEGRAM_CHAT_ID, 'text': text, 'parse_mode': 'Markdown'},
            timeout=10,
        )
    except Exception as e:
        logger.warning(f"Telegram alert failed: {e}")


def alert_trade_open(symbol, direction, entry, tp, sl, lot_size, conf, regime, ict_score):
    _send(
        f"🔔 *TRADE OPEN*\n"
        f"Symbol: `{symbol}` | {direction}\n"
        f"Entry: `{entry:.2f}` | TP: `{tp:.2f}` | SL: `{sl:.2f}`\n"
        f"Lot: `{lot_size}` | Conf: `{conf:.0%}`\n"
        f"Regime: `{'Trending' if regime else 'Ranging'}` | ICT score: `{ict_score}/5`"
    )


def alert_trade_close(symbol, direction, entry, exit_price, result, r_multiple):
    emoji = "✅" if result == 1 else "❌"
    _send(
        f"{emoji} *TRADE CLOSED*\n"
        f"`{symbol}` | {direction}\n"
        f"Entry: `{entry:.2f}` → Exit: `{exit_price:.2f}`\n"
        f"{'WIN' if result == 1 else 'LOSS'} | `{r_multiple:+.2f}R`"
    )


def alert_partial_close(symbol, direction, price, r_so_far):
    _send(
        f"🔒 *PARTIAL CLOSE* — `{symbol}` {direction}\n"
        f"50% closed @ `{price:.2f}` | `{r_so_far:.2f}R` locked in\n"
        f"SL moved to breakeven."
    )


def alert_halt(reason: str):
    _send(f"🛑 *TRADING HALTED*\n{reason}")


def alert_resume():
    _send("✅ *Trading resumed* (daily reset).")


def alert_confidence_drift(symbol, avg_conf):
    _send(
        f"⚠️ *CONFIDENCE DRIFT* — `{symbol}`\n"
        f"Rolling avg confidence: `{avg_conf:.0%}` (floor: 58%)\n"
        f"Consider retraining the model."
    )


def alert_startup(symbols, balance, deployable):
    _send(
        f"🚀 *Bot started*\n"
        f"Symbols: `{', '.join(symbols)}`\n"
        f"Balance: `${balance:,.2f}`\n"
        f"Walk-forward: `{'PASSED ✅' if deployable else 'FAILED ⚠️ — running anyway'}`"
    )


def alert_daily_summary(balance, daily_pnl, win_rate, total_trades, max_dd):
    direction = "📈" if daily_pnl >= 0 else "📉"
    _send(
        f"{direction} *Daily Summary*\n"
        f"Balance: `${balance:,.2f}` | P&L: `${daily_pnl:+,.2f}`\n"
        f"Trades: `{total_trades}` | Win%: `{win_rate:.0%}`\n"
        f"Max DD today: `{max_dd:.1%}`"
    )
