import os
from dotenv import load_dotenv

load_dotenv()

# ==================== EXCHANGE ====================
API_KEY    = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
TESTNET    = True   # ENABLED: Pointing to modern Demo-FAPI Dashboard

# ==================== SYMBOLS ====================
SYMBOLS = ['DOT/USDT', 'LINK/USDT', 'NEAR/USDT']

# ==================== TIMEFRAMES ====================
TF_SIGNAL = '15m'
TF_BIAS   = '1h'
SEQ_LEN   = 60   # LSTM lookback window

# ==================== RISK ====================
RISK_PERCENT      = 0.02   # 2% account risk per trade
MAX_TRADES_TOTAL  = 3      # One per symbol (DOT, LINK, NEAR)
DAILY_LOSS_LIMIT  = 0.04   # Increased for 2% risk context
DD_CIRCUIT_BREAK  = 0.10   # Halt at 10% drawdown
COOLDOWN_MINUTES  = 60     # Extended cooldown for structural resets
MIN_RR            = 2.0    # Targeting 3.0:1 base RR
TP_MULT           = 3.6    # Real-world TP: 3.6x ATR
SL_MULT           = 1.2    # Real-world SL: 1.2x ATR
PARTIAL_CLOSE_R   = 1.0    # Target 1: 1.0R profit
PARTIAL_CLOSE_AMT = 0.2    # Sell 20% at Target 1
BREAKEVEN_TRIGGER_R = 1.4  # Move SL to Entry after 1.4R

# ==================== STRATEGY: HYBRID PULSE ====================
USE_RECLAIM_CONFIRM = True  # Wait for price to close back above/below EMA after touch
USE_PULSE_FILTER    = True  # Only enter if Bollinger Bands are expanding
USE_4H_FILTER       = False # Too lagging for 15m signal engine

# Adaptive Trailing Exits
USE_ADAPTIVE_EXIT   = False # FIXED: Verified TP/SL is superior to trailing for ICT
TRAILING_TRIGGER_R  = 1.6   # (Unused in Pure ICT mode)
BREAKEVEN_FIXED_RR  = 1.4   # SL moves to Entry at 1.4R
TRAILING_STOP_ATR   = 1.0   # Trailing distance in ATR units

# ==================== SESSION FILTER ====================
# London open through NY close (highest quality moves): 08:00–20:00 UTC
SESSION_START_UTC = 8
SESSION_END_UTC   = 20

# ==================== MODEL ====================
TRAIN_EPOCHS        = 100
CONFIDENCE_THRESH   = 0.68   # Min ensemble confidence to enter a trade
CONFIDENCE_FLOOR    = 0.58   # Alert if rolling avg confidence drops below this
REPLAY_BUFFER_SIZE  = 100    # Max samples in online-learning replay buffer
REPLAY_BATCH_SIZE   = 32     # Mini-batch size for online updates

# ==================== WALK-FORWARD ====================
WF_TRAIN_MONTHS = 8    # Training window size
WF_TEST_MONTHS  = 1    # Out-of-sample test window size
WF_MIN_SHARPE   = 0.8  # Minimum Sharpe across all windows to deploy

ADX_MIN           = 20     # Relaxed from 25 for Triple Threat volatility
RSI_LONG_MAX      = 52     # Sync with backtest
RSI_SHORT_MIN     = 48     # Sync with backtest
LIMIT_OFFSET_ATR  = 0.15   # Limit order placed inside signal candle (better fill)
LIMIT_TIMEOUT_MIN = 7      # Cancel unfilled limit orders after N minutes
FEE_MAKER         = 0.0002
FEE_TAKER         = 0.0004

# ==================== MISC ====================
BALANCE_REFRESH_CANDLES = 10   # Refresh balance every N candles to avoid rate limits
LOOP_SLEEP_SECONDS      = 30

# ==================== PATHS ====================
MODEL_DIR  = 'models'
LOG_DIR    = 'logs'
PERF_LOG   = 'logs/performance.csv'
EQUITY_LOG = 'logs/equity.csv'
CONF_LOG   = 'logs/confidence.csv'

# ==================== TELEGRAM ====================
# Set via environment variables or fill in directly
TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_TOKEN',   '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
