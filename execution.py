import time
import logging
from config import LIMIT_OFFSET_ATR, LIMIT_TIMEOUT_MIN

logger = logging.getLogger(__name__)


# ==================== OPEN ====================

def place_limit_entry(exchange, symbol, side, lot_size, price, atr):
    """
    Places a limit order slightly inside the signal candle.
    - Long: limit below current price  → better fill, lower risk
    - Short: limit above current price → better fill, lower risk

    Returns (order_id, limit_price) or (None, price) on failure.
    """
    offset = round(LIMIT_OFFSET_ATR * atr, 2)

    if side == 'buy':
        limit_price = round(price - offset, 2)
    else:
        limit_price = round(price + offset, 2)

    try:
        order = exchange.create_limit_order(symbol, side, lot_size, limit_price)
        order_id = order['id']
        logger.info(f"Limit {side.upper()} placed | {lot_size} {symbol} @ {limit_price:.2f}")
        return order_id, limit_price
    except Exception as e:
        logger.error(f"Limit order failed ({symbol} {side}): {e}")
        return None, price


def wait_for_fill(exchange, symbol, order_id):
    """
    Polls until the limit order is filled or LIMIT_TIMEOUT_MIN elapses.
    Returns True if filled, False if cancelled/timed-out.
    """
    timeout = LIMIT_TIMEOUT_MIN * 60
    elapsed = 0
    poll    = 15   # seconds between checks

    while elapsed < timeout:
        time.sleep(poll)
        elapsed += poll
        try:
            order = exchange.fetch_order(order_id, symbol)
            status = order.get('status', '')
            if status == 'closed':
                logger.info(f"Order {order_id} filled.")
                return True
            if status in ('canceled', 'expired', 'rejected'):
                logger.info(f"Order {order_id} not filled (status: {status}).")
                return False
        except Exception as e:
            logger.warning(f"Order poll error: {e}")

    # Timed out — cancel
    try:
        exchange.cancel_order(order_id, symbol)
        logger.info(f"Order {order_id} cancelled (timeout after {LIMIT_TIMEOUT_MIN}m).")
    except Exception as e:
        logger.warning(f"Cancel failed for {order_id}: {e}")

    return False


# ==================== CLOSE ====================

def close_partial(exchange, symbol, trade_type, lot_size, fraction=0.5):
    """
    Market-close a fraction of an open position.
    Returns the quantity closed.
    """
    qty = round(lot_size * fraction, 3)
    if qty <= 0:
        return 0.0

    side = 'sell' if trade_type == 'LONG' else 'buy'
    try:
        exchange.create_market_order(symbol, side, qty)
        logger.info(f"Partial close {fraction*100:.0f}% | {qty} {symbol}")
        return qty
    except Exception as e:
        logger.error(f"Partial close failed ({symbol}): {e}")
        return 0.0


def close_full(exchange, symbol, trade_type, lot_size):
    """
    Market-close the entire remaining position.
    Returns True on success.
    """
    side = 'sell' if trade_type == 'LONG' else 'buy'
    try:
        exchange.create_market_order(symbol, side, lot_size)
        logger.info(f"Full close {trade_type} | {lot_size} {symbol}")
        return True
    except Exception as e:
        logger.error(f"Full close failed ({symbol}): {e}")
        return False
