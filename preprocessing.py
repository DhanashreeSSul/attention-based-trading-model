import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_bollinger_bands(series, window=20):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return upper, lower, (series - lower) / (upper - lower + 1e-9)  # %B


def add_indicators(df):
    df['MA5']  = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['EMA']  = df['Close'].ewm(span=10).mean()
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']
    df['Volatility']  = df['Close'].rolling(10).std()
    df['Momentum']    = df['Close'] - df['Close'].shift(5)
    df['RSI']         = compute_rsi(df['Close'])
    df['BB_upper'], df['BB_lower'], df['BB_pct'] = compute_bollinger_bands(df['Close'])
    df['Return1']  = df['Close'].pct_change(1)
    df['Return5']  = df['Close'].pct_change(5)
    df['VolRatio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df.fillna(0, inplace=True)
    return df


def create_labels(df, news_df=None):
    """Use DJIA Label from news CSV if available, else compute from price."""
    if news_df is not None and 'Label' in news_df.columns:
        # Map DJIA labels directly (0=down, 1=up) — ground truth
        news_df = news_df[['Date', 'Label']].copy()
        news_df['Date'] = pd.to_datetime(news_df['Date'])
        df = df.merge(news_df, on='Date', how='left')
        df['Target'] = df['Label'].fillna(method='ffill').astype(int)
    else:
        future_return = (df['Close'].shift(-1) - df['Close']) / df['Close']
        df['Target'] = (future_return > 0).astype(int)

    df = df.dropna(subset=['Target'])
    return df


def normalize_features(df, feature_cols):
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df


def merge_data(stock_df, news_df):
    # Use ALL 25 headlines for richer context
    top_cols = [c for c in news_df.columns if c.startswith('Top')]
    news_df[top_cols] = news_df[top_cols].fillna("")

    # Concatenate top headlines (up to 5 most informative)
    news_df['News'] = news_df[top_cols[:5]].apply(
        lambda row: ' [SEP] '.join([str(x).strip("b'\"") for x in row if str(x).strip("b'\"")])[:512],
        axis=1
    )

    news_df['Date'] = pd.to_datetime(news_df['Date'])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    merged = pd.merge(stock_df, news_df[['Date', 'News', 'Label']], on='Date', how='inner')
    merged['News'] = merged['News'].fillna("")

    return merged
