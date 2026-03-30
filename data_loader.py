import yfinance as yf
import pandas as pd


def load_stock_data(ticker="AAPL", start="2015-01-01", end="2023-01-01"):
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError("Stock data not downloaded.")
    # 🔥 FIX: Flatten MultiIndex columns
    if isinstance(df.columns, tuple) or hasattr(df.columns, 'levels'):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.reset_index(inplace=True)
    return df


def load_news_data(path):
    df = pd.read_csv("Combined_News_DJIA.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df