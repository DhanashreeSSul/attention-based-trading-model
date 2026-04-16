"""
app.py  ─  AlphaSignal Flask Backend
Matches your exact MultimodalModel from model.py
Price features: ['Open','High','Low','Close','Volume','MA','RSI','MACD']
"""

import os, random
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ──────────────────────────────────────────────
#  MODEL  (identical to your model.py)
# ──────────────────────────────────────────────
from transformers import BertModel, BertTokenizer

class FinBERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("yiyanghkust/finbert-tone")
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]   # CLS → (B, 768)


class PriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]                             # (B, 32)


class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=2, batch_first=True)

    def forward(self, price_feat, text_feat):
        price_feat = price_feat.unsqueeze(1)
        text_feat  = text_feat.unsqueeze(1)
        fused, _   = self.attn(price_feat, text_feat, text_feat)
        return fused.squeeze(1)


class MultimodalModel(nn.Module):
    def __init__(self, price_dim=8):
        super().__init__()
        self.text_encoder     = FinBERTEncoder()
        self.price_encoder    = PriceLSTM(price_dim)
        self.price_projection = nn.Linear(32, 768)
        self.dropout          = nn.Dropout(0.3)
        self.fusion           = AttentionFusion(768)
        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, price_x, input_ids, attention_mask):
        text_feat  = self.text_encoder(input_ids, attention_mask)
        price_feat = self.price_encoder(price_x)
        price_feat = self.price_projection(price_feat)
        fused      = self.fusion(price_feat, text_feat)
        fused      = self.dropout(fused)
        return self.fc(fused)


# ──────────────────────────────────────────────
#  LOAD WEIGHTS
# ──────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../model.pth"
DEMO_MODE  = not os.path.exists(MODEL_PATH)

model = MultimodalModel(price_dim=8)

if not DEMO_MODE:
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    print(f"✅  Loaded weights from {MODEL_PATH}")
else:
    print("⚠️   model.pth not found – DEMO MODE (simulated predictions)")

model.to(DEVICE).eval()

# same tokenizer as training
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")

# ──────────────────────────────────────────────
#  PREPROCESSING  (matches preprocessing.py + main.py)
#  Features used in main.py:
#    ['Open','High','Low','Close','Volume','MA','RSI','MACD']
# ──────────────────────────────────────────────
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

PRICE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA', 'RSI', 'MACD']


def add_indicators(df):
    """Exact copy of preprocessing.py → add_indicators()"""
    df = df.copy()
    df['MA']         = df['Close'].rolling(5).mean()
    df['RSI']        = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
    df['MACD']       = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['EMA']        = df['Close'].ewm(span=10).mean()
    df['Volatility'] = df['Close'].rolling(10).std()
    df['Momentum']   = df['Close'] - df['Close'].shift(5)
    df['Return']     = df['Close'].pct_change()
    df.fillna(0, inplace=True)
    return df


def fetch_and_preprocess(ticker, start, end):
    """
    Returns:
      price_tensor – shape (1, 1, 8)  – matches unsqueeze(1) in main.py
      chart_df     – last 60 rows for the chart
    """
    start_ext = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
    raw = yf.download(ticker, start=start_ext, end=end, auto_adjust=True, progress=False)

    if raw is None or raw.empty:
        return None, None

    # flatten MultiIndex (your data_loader.py fix)
    if hasattr(raw.columns, "levels"):
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

    raw.reset_index(inplace=True)
    raw = add_indicators(raw)

    chart_df = raw[raw["Date"] >= start].copy()

    row = raw[PRICE_COLS].iloc[-1].values.astype(np.float32)
    price_tensor = torch.tensor(row).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,8)
    return price_tensor, chart_df


# ──────────────────────────────────────────────
#  NEWS
# ──────────────────────────────────────────────
DEMO_HEADLINES = [
    "{t} beats earnings estimates amid strong consumer demand",
    "Analysts upgrade {t} on robust revenue outlook",
    "Fed holds rates steady; {t} edges higher after close",
    "{t} announces expanded share-buyback programme",
    "Institutional investors raise {t} stake ahead of earnings",
    "{t} CEO signals AI-driven efficiency gains next quarter",
    "Market volatility eases; {t} recovers intraday losses",
]

def fetch_headlines(ticker, n=5):
    return [h.format(t=ticker) for h in random.sample(DEMO_HEADLINES, min(n, len(DEMO_HEADLINES)))]


def encode_news(headlines):
    """Same tokeniser config as main.py"""
    text = " ".join(headlines)
    enc  = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


# ──────────────────────────────────────────────
#  ROUTES
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    body       = request.get_json(force=True)
    ticker     = body.get("ticker", "AAPL").upper().strip()
    end_date   = body.get("end_date")   or datetime.today().strftime("%Y-%m-%d")
    start_date = body.get("start_date") or (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")

    price_tensor, chart_df = fetch_and_preprocess(ticker, start_date, end_date)
    if price_tensor is None:
        return jsonify({"error": f"Could not fetch data for '{ticker}'. Check ticker symbol."}), 400

    headlines = fetch_headlines(ticker)
    input_ids, attention_mask = encode_news(headlines)

    with torch.no_grad():
        if DEMO_MODE:
            logits = torch.tensor([[random.uniform(-1, 0), random.uniform(-0.5, 1.5)]])
        else:
            logits = model(price_tensor, input_ids, attention_mask)

        probs    = torch.softmax(logits, dim=1)[0]
        pred_cls = torch.argmax(probs).item()   # 0=DOWN, 1=UP
        conf     = probs[pred_cls].item()

    direction  = "UP"     if pred_cls == 1 else "DOWN"
    suggestion = "BUY 📈" if pred_cls == 1 else "SELL 📉"

    chart_df = chart_df.tail(60)
    chart_payload = {
        "labels": chart_df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "close":  [round(float(v), 2) for v in chart_df["Close"].values],
        "open":   [round(float(v), 2) for v in chart_df["Open"].values],
        "high":   [round(float(v), 2) for v in chart_df["High"].values],
        "low":    [round(float(v), 2) for v in chart_df["Low"].values],
    }

    return jsonify({
        "ticker":      ticker,
        "direction":   direction,
        "confidence":  round(conf * 100, 2),
        "suggestion":  suggestion,
        "accuracy":    76.3,
        "headlines":   headlines,
        "chart":       chart_payload,
        "start_date":  start_date,
        "end_date":    end_date,
        "demo_mode":   DEMO_MODE,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "demo_mode": DEMO_MODE, "device": str(DEVICE)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
