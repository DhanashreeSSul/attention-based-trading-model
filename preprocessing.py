import pandas as pd
import numpy as np


def add_indicators(df):
    df['MA'] = df['Close'].rolling(5).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['EMA'] = df['Close'].ewm(span=10).mean()
    df['Volatility'] = df['Close'].rolling(10).std()
    df.fillna(0, inplace=True)
    return df


def create_labels(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df


def merge_data(stock_df, news_df):
    # Identify news columns
    news_cols = [f"Top{i}" for i in range(1, 4)]  # Top1–Top3 only
    # Replace NaN with empty string FIRST
    news_df[news_cols] = news_df[news_cols].fillna("")
    # Convert everything safely to string and join
    news_df['News'] = news_df[news_cols].apply(
        lambda row: ' '.join([str(x) for x in row if str(x) != "nan"]),
        axis=1
    )
    # Convert dates
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    # Merge datasets
    merged = pd.merge(stock_df, news_df[['Date', 'News']], on='Date', how='left')
    # Fill missing news after merge
    merged['News'] = merged['News'].fillna("")
    return merged
