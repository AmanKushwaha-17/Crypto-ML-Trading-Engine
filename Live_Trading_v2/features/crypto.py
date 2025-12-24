import pandas as pd
import numpy as np

# ---------- Core feature builder ----------

def build_crypto_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw futures OHLCV dataframe with:
        ['open_time','open','high','low','close','volume','taker_buy_base','funding_rate', ...]
    and returns a dataframe with added technical & regime features.
    """

    # 1) Ensure numeric dtypes
    for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base', 'funding_rate']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2) Time handling
    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df.sort_values('open_time').reset_index(drop=True)

    # --- CRITICAL FIX: Create lagged prices for calculations ---
    # At decision time t, we only know prices up to t-1
    df['close_lag1'] = df['close'].shift(1)
    df['high_lag1'] = df['high'].shift(1)
    df['low_lag1'] = df['low'].shift(1)
    df['volume_lag1'] = df['volume'].shift(1)

    # 3) Log returns - FIXED: calculate on lagged close
    df['log_returns'] = np.log(df['close_lag1'] / df['close_lag1'].shift(1))

    # ---------- Time features ----------
    df['day_of_week'] = df['open_time'].dt.dayofweek  # Monday=0
    df['hour'] = df['open_time'].dt.hour
    df['day_of_month'] = df['open_time'].dt.day
    df['month'] = df['open_time'].dt.month
    df['quarter'] = df['open_time'].dt.quarter

    # Cyclical encodings+
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # ---------- RSI ---------- FIXED
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['rsi_14'] = calculate_rsi(df['close_lag1'], 14)
    df['rsi_7'] = calculate_rsi(df['close_lag1'], 7)

    # ---------- MACD ---------- FIXED
    ema_12 = df['close_lag1'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close_lag1'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ---------- Bollinger Bands ---------- FIXED
    df['bb_middle'] = df['close_lag1'].rolling(20).mean()
    bb_std = df['close_lag1'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
    df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # Avoid division by zero in bb_position
    bb_denom = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    df['bb_position'] = (df['close_lag1'] - df['bb_lower']) / bb_denom

    # ---------- Stochastic Oscillator ---------- FIXED
    low_14 = df['low_lag1'].rolling(14).min()
    high_14 = df['high_lag1'].rolling(14).max()
    denom = (high_14 - low_14).replace(0, np.nan)
    df['stoch_k'] = 100 * (df['close_lag1'] - low_14) / denom
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # ---------- ATR (volatility) ---------- FIXED
    high_low = df['high_lag1'] - df['low_lag1']
    high_close = (df['high_lag1'] - df['close_lag1'].shift()).abs()
    low_close = (df['low_lag1'] - df['close_lag1'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / df["close_lag1"]

    # ---------- OBV ---------- FIXED
    df['obv'] = (np.sign(df['close_lag1'].diff()) * df['volume_lag1']).fillna(0).cumsum()
    df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()

    # ---------- Money Flow Index (MFI) ---------- FIXED
    typical_price = (df['high_lag1'] + df['low_lag1'] + df['close_lag1']) / 3
    raw_money_flow = typical_price * df['volume_lag1']
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()

    # Avoid division by zero
    money_ratio = positive_flow / negative_flow.replace(0, np.nan)
    df['mfi'] = 100 - (100 / (1 + money_ratio))

    # ---------- Volatility features ---------- FIXED
    df['volatility_10'] = df['log_returns'].rolling(10).std()
    df['volatility_24'] = df['log_returns'].rolling(24).std()
    df['volatility_168'] = df['log_returns'].rolling(168).std()
    df['vol_ratio'] = df['volatility_10'] / df['volatility_168']

    # ---------- Volume features ---------- FIXED
    df['volume_ma_24'] = df['volume_lag1'].rolling(24).mean()
    df['volume_ma_168'] = df['volume_lag1'].rolling(168).mean()
    df['volume_ratio'] = df['volume_lag1'] / df['volume_ma_24']
    df['volume_trend'] = df['volume_ma_24'] / df['volume_ma_168']

    # ---------- Buy pressure ---------- FIXED
    df['buy_pressure'] = df['taker_buy_base'] / df['volume_lag1'].replace(0, np.nan)
    df['buy_pressure_ma'] = df['buy_pressure'].rolling(10).mean()
    df['buy_strength'] = df['buy_pressure'] - df['buy_pressure_ma']

    # ---------- Price momentum ---------- FIXED
    df['roc_4'] = (df['close_lag1'] - df['close_lag1'].shift(4)) / df['close_lag1'].shift(4)
    df['roc_12'] = (df['close_lag1'] - df['close_lag1'].shift(12)) / df['close_lag1'].shift(12)
    df['roc_24'] = (df['close_lag1'] - df['close_lag1'].shift(24)) / df['close_lag1'].shift(24)

    # ---------- Moving averages / trend ---------- FIXED
    df['sma_24'] = df['close_lag1'].rolling(24).mean()
    df['sma_168'] = df['close_lag1'].rolling(168).mean()
    df['price_to_sma24'] = (df['close_lag1'] - df['sma_24']) / df['sma_24']
    df['sma_cross'] = (df['sma_24'] > df['sma_168']).astype(int)

    # In calculate_dynamic_leverage_strategy:
    df["ema_fast"] = df["close_lag1"].ewm(span=50, adjust=False).mean()
    df["ema_slow"] = df["close_lag1"].ewm(span=200, adjust=False).mean()

    # High-low range
    df['hl_range'] = (df['high_lag1'] - df['low_lag1']) / df['close_lag1']
    df['hl_range_ma'] = df['hl_range'].rolling(24).mean()

    # Return lags
    for i in [1, 2, 3, 4, 6, 8, 12, 24]:
        df[f'return_lag_{i}'] = df['log_returns'].shift(i)

    # Trend regime (Bull / Bear / Neutral) using SMA200
    df['sma_200'] = df['close_lag1'].rolling(200).mean()
    df['trend_regime'] = np.where(
        df['close_lag1'] > df['sma_200'], 1,
        np.where(df['close_lag1'] < df['sma_200'], -1, 0)
    )

    # Clean up: remove temporary lagged columns
    df = df.drop(columns=['close_lag1', 'high_lag1', 'low_lag1', 'volume_lag1'])

    return df


# ---------- Helper: load + build in one shot ----------

def load_and_build(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = build_crypto_features(df)
    return df