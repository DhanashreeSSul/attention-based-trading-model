# AlphaSignal — UI for Your Multimodal Trading Model

Attention-Based Multimodal Deep Learning UI  
**FinBERT + LSTM + Attention Fusion → Stock Movement Prediction**

---

## Folder Structure

```
trading-ui/
├── app.py                  ← Flask backend (main server)
├── model.pth               ← YOUR trained model (place here!)
├── requirements.txt
├── templates/
│   └── index.html          ← Frontend HTML
└── static/
    ├── style.css
    └── script.js
```

---

## Step 1 — Place Your Trained Model

Copy your saved `model.pth` into the `trading-ui/` folder:

```bash
cp /path/to/your/model.pth trading-ui/model.pth
```

> If you don't have model.pth yet, the app runs in **Demo Mode**
> (shows simulated predictions so you can test the UI).

To save your model after training, add this to `train.py`:

```python
torch.save(model.state_dict(), "model.pth")
# OR with metadata:
torch.save({"model_state_dict": model.state_dict(), "accuracy": 76.3}, "model.pth")
```

---

## Step 2 — Install Dependencies

```bash
cd trading-ui
pip install -r requirements.txt
```

---

## Step 3 — Run the Flask Server

```bash
python app.py
```

You should see:
```
✅  Loaded weights from model.pth    ← if model.pth exists
 * Running on http://127.0.0.1:5000
```

---

## Step 4 — Open the UI

Open your browser and go to:

```
http://localhost:5000
```

---

## How It Works

| Component        | Detail |
|-----------------|--------|
| **Stock Data**  | Downloaded live from Yahoo Finance via `yfinance` |
| **Features**    | `Open, High, Low, Close, Volume, MA, RSI, MACD` (matches `main.py`) |
| **Text**        | Sample headlines → tokenised with `yiyanghkust/finbert-tone` |
| **Inference**   | Same `MultimodalModel` class as `model.py` |
| **Output**      | UP/DOWN + confidence % + BUY/SELL suggestion |

---

## API Endpoint

**POST** `/predict`

```json
{
  "ticker":     "AAPL",
  "start_date": "2024-01-01",
  "end_date":   "2024-12-31"
}
```

Response:
```json
{
  "ticker":     "AAPL",
  "direction":  "UP",
  "confidence": 73.4,
  "suggestion": "BUY 📈",
  "accuracy":   76.3,
  "headlines":  ["..."],
  "chart":      { "labels": [...], "close": [...] },
  "demo_mode":  false
}
```

---

## Using Real News (Optional Upgrade)

Replace `fetch_headlines()` in `app.py` with a real news API:

**Finnhub (free tier):**
```python
import requests
def fetch_headlines(ticker, n=5):
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2024-01-01&to=2024-12-31&token=YOUR_KEY"
    data = requests.get(url).json()
    return [item["headline"] for item in data[:n]]
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `model.pth not found` | App runs in Demo Mode — place `model.pth` in `trading-ui/` |
| `Could not fetch data` | Check ticker symbol (use Yahoo Finance format e.g. `RELIANCE.NS`) |
| `Port 5000 in use` | Change port in `app.py`: `app.run(port=5001)` |
| FinBERT download slow | First run downloads ~440MB model — subsequent runs are cached |
