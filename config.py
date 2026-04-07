import os
from dotenv import load_dotenv

load_dotenv()

# ==================== EXCHANGE ====================
API_KEY    = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
TESTNET    = True

# ==================== SYMBOLS ====================
SYMBOLS = ['BTC/USDT', 'ETH/USDT']

# ==================== TIMEFRAMES ====================
TF_SIGNAL = '15m'
TF_BIAS   = '1h'
SEQ_LEN   = 60   # LSTM lookback window

# ==================== RISK ====================
RISK_PERCENT      = 0.01   # 1% account risk per trade
MAX_TRADES_TOTAL  = 2      # Max concurrent open trades across all symbols
DAILY_LOSS_LIMIT  = 0.02   # Halt trading at 2% daily drawdown
DD_CIRCUIT_BREAK  = 0.06   # Halt trading at 6% drawdown from equity peak
COOLDOWN_MINUTES  = 45     # Wait after a trade closes before re-entering same symbol
MIN_RR            = 0.5    # Minimum reward:risk ratio — skip trade if not met
PARTIAL_CLOSE_R   = 1.0    # Close 50% of position when trade reaches 1R profit

# ==================== SESSION FILTER ====================
# London/NY overlap (highest quality moves): 13:00–17:00 UTC
SESSION_START_UTC = 13
SESSION_END_UTC   = 17

# ==================== MODEL ====================
TRAIN_EPOCHS        = 50
CONFIDENCE_THRESH   = 0.68   # Min ensemble confidence to enter a trade
CONFIDENCE_FLOOR    = 0.58   # Alert if rolling avg confidence drops below this
REPLAY_BUFFER_SIZE  = 100    # Max samples in online-learning replay buffer
REPLAY_BATCH_SIZE   = 32     # Mini-batch size for online updates

# ==================== WALK-FORWARD ====================
WF_TRAIN_MONTHS = 8    # Training window size
WF_TEST_MONTHS  = 1    # Out-of-sample test window size
WF_MIN_SHARPE   = 0.8  # Minimum Sharpe across all windows to deploy

# ==================== EXECUTION ====================
LIMIT_OFFSET_ATR  = 0.15   # Limit order placed N * ATR inside signal candle (better fill)
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
