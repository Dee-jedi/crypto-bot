import pandas as pd
import numpy as np
from data_feed import compute_cvd
from config import SESSION_START_UTC, SESSION_END_UTC
import logging

logger = logging.getLogger(__name__)

# All features fed to the model. Order matters for scaler persistence.
FEAT_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'ATR', 'RSI', 'EMA9', 'EMA21', 'EMA20', 'EMA50', 'EMA200',
    'ADX', 'PlusDI', 'MinusDI',
    'ATR_Pct', 'EMA50_Slope', 'Momentum',
    'CVD_Delta', 'OI_Delta', 'FundingRate',
    'Price_EMA50_Dist', 'Price_EMA200_Dist',
    'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Pct',
    'VWAP',
    'InSession', 'Regime',
]


# ==================== CORE INDICATORS ====================

def _atr(df, period=14):
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _rsi(df, period=14):
    delta    = df['Close'].diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _adx(df, period=14):
    high, low = df['High'], df['Low']

    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    atr      = _atr(df, period)
    plus_di  = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean()  / (atr + 1e-10)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-10)

    dx  = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx, plus_di, minus_di


def _atr_percentile(atr_series, period=100):
    """
    Rank current ATR vs last N bars. 0 = most compressed, 1 = most expanded.
    Compressed (low percentile) → breakout pending.
    Expanded (high percentile)  → mean-reversion more likely.
    """
    return atr_series.rolling(period).rank(pct=True)


def _bollinger_bands(df, period=20, std_dev=2.0):
    """
    Bollinger Bands: SMA +/- N standard deviations.
    Price outside bands = statistically overextended.
    """
    sma = df['Close'].rolling(period).mean()
    std = df['Close'].rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = (upper - lower) / (sma + 1e-10)  # Normalized band width
    pct   = (df['Close'] - lower) / (upper - lower + 1e-10)  # 0=lower band, 1=upper band
    return upper, lower, width, pct


def _vwap(df):
    """
    Session-based VWAP proxy using rolling 96-bar window (24h on 15m).
    Acts as institutional fair value level for mean-reversion.
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    vol = df['Volume']
    cum_tp_vol = (typical_price * vol).rolling(96).sum()
    cum_vol    = vol.rolling(96).sum()
    return cum_tp_vol / (cum_vol + 1e-10)


# ==================== REGIME ====================

def classify_regime(row):
    """
    Rule-based regime detection. No separate model needed.
    1 = trending  : ADX > 25 and volatility expanding
    0 = ranging   : everything else
    """
    return int(row['ADX'] > 25 and row['ATR_Pct'] > 0.45)


# ==================== BUILD FEATURES ====================

def build_features(df, funding_rate=0.0, oi_df=None):
    """
    Computes all features on a raw OHLCV DataFrame.

    Args:
        df           : Raw OHLCV DataFrame (time index, Open/High/Low/Close/Volume).
        funding_rate : Scalar. Current funding rate from exchange.
        oi_df        : Optional DataFrame with 'OI' column, same time index as df.

    Returns:
        DataFrame with all FEAT_COLS populated and NaN rows dropped.
    """
    d = df.copy()

    # --- Price-based ---
    d['ATR']         = _atr(d)
    d['RSI']         = _rsi(d)
    d['EMA9']        = d['Close'].ewm(span=9).mean()
    d['EMA21']       = d['Close'].ewm(span=21).mean()
    d['EMA20']       = d['Close'].ewm(span=20).mean()
    d['EMA50']       = d['Close'].ewm(span=50).mean()
    d['EMA200']      = d['Close'].ewm(span=200).mean()
    d['Momentum']    = d['Close'].pct_change(5)
    d['EMA50_Slope'] = d['EMA50'].pct_change(3)

    # --- ADX / DI ---
    d['ADX'], d['PlusDI'], d['MinusDI'] = _adx(d)

    # --- ATR Percentile (volatility regime) ---
    d['ATR_Pct'] = _atr_percentile(d['ATR'])

    # --- Distance from EMAs (normalised by ATR) ---
    d['Price_EMA50_Dist']  = (d['Close'] - d['EMA50'])  / (d['ATR'] + 1e-10)
    d['Price_EMA200_Dist'] = (d['Close'] - d['EMA200']) / (d['ATR'] + 1e-10)

    # --- Bollinger Bands ---
    d['BB_Upper'], d['BB_Lower'], d['BB_Width'], d['BB_Pct'] = _bollinger_bands(d)

    # --- VWAP proxy ---
    d['VWAP'] = _vwap(d)

    # --- CVD ---
    cvd          = compute_cvd(d)
    d['CVD_Delta'] = cvd.diff(3)   # 3-candle momentum of CVD

    # --- Open Interest ---
    if oi_df is not None and not oi_df.empty:
        oi_aligned  = oi_df['OI'].reindex(d.index, method='ffill')
        d['OI_Delta'] = oi_aligned.pct_change(3).fillna(0.0)
    else:
        d['OI_Delta'] = 0.0

    # --- Funding rate (broadcast scalar) ---
    d['FundingRate'] = funding_rate

    # --- Session filter ---
    d['InSession'] = d.index.hour.isin(
        range(SESSION_START_UTC, SESSION_END_UTC)
    ).astype(float)

    # --- Regime label ---
    d.dropna(inplace=True)
    d['Regime'] = d.apply(classify_regime, axis=1).astype(float)

    return d


# ==================== BIAS (higher timeframe) ====================

def htf_bias(df_1h):
    """
    Returns 'BULL', 'BEAR', or 'NEUTRAL' based on multi-EMA alignment.
    All three EMAs must be stacked in order for a directional bias.
    NEUTRAL = mixed/transitional → skip trading.
    """
    ema20  = df_1h['EMA20'].iloc[-1]
    ema50  = df_1h['EMA50'].iloc[-1]
    ema200 = df_1h['EMA200'].iloc[-1]
    if ema20 > ema50 > ema200:
        return 'BULL'
    elif ema20 < ema50 < ema200:
        return 'BEAR'
    return 'NEUTRAL'
