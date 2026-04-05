import torch
from transformers import AutoTokenizer
from data_loader import load_stock_data, load_news_data
from preprocessing import add_indicators, create_labels, merge_data
from model import MultimodalModel
from train import train_model
from evaluate import evaluate
from sklearn.model_selection import train_test_split

# Load Data
stock = load_stock_data()
news = load_news_data("Comined_News_DJIA.csv")

stock = add_indicators(stock)
stock = create_labels(stock)
merged = merge_data(stock, news)
merged = merged.sort_values('Date')
merged = merged.ffill()
merged = merged[:1000] 

print(news.columns)
# Tokenizer
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")

texts = merged['News'].tolist()
enc = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,  
    return_tensors="pt"
)

# Price features
price_features = merged[['Open','High','Low','Close','Volume','MA','RSI','MACD']].values
price_tensor = torch.tensor(price_features, dtype=torch.float32).unsqueeze(1)
labels = torch.tensor(merged['Target'].values, dtype=torch.long)

# Model
model = MultimodalModel(price_dim=price_tensor.shape[-1])

from sklearn.model_selection import train_test_split

X_price_train, X_price_test, ids_train, ids_test, mask_train, mask_test, y_train, y_test = train_test_split(
    price_tensor, enc['input_ids'], enc['attention_mask'], labels,
    test_size=0.2,
    random_state=42
)

train_model(model,
            X_price_train, ids_train, mask_train, y_train,
            X_price_test, ids_test, mask_test, y_test)
# Train
# train_model(model, X_price_train, ids_train, mask_train, y_train)
metrics = evaluate(model, X_price_test, ids_test, mask_test, y_test)
print(metrics)


# =============================
# Trading Simulation (add in main)
# =============================

def simulate_trading(preds, prices):
    cash = 10000
    shares = 0

    for i in range(len(preds)):
        if preds[i] == 1:  # BUY
            shares = cash / prices[i]
            cash = 0
        elif preds[i] == 0 and shares > 0:  # SELL
            cash = shares * prices[i]
            shares = 0

    return cash