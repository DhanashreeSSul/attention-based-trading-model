import torch
from transformers import BertTokenizer
from data_loader import load_stock_data, load_news_data
from preprocessing import add_indicators, create_labels, merge_data, normalize_features
from model import MultimodalModel
from train import train_model
from evaluate import evaluate
from utils import create_sequences
from sklearn.model_selection import train_test_split

print("=" * 60)
print("STOCK PREDICTION — IMPROVED PIPELINE")
print("=" * 60)

# ── LOAD ──────────────────────────────────────────────────────
stock = load_stock_data(ticker="^DJI", start="2008-01-01", end="2016-07-01")
news  = load_news_data("Combined_News_DJIA.csv")

print(f"Stock rows: {len(stock)} | News rows: {len(news)}")

# ── FEATURES & LABELS ────────────────────────────────────────
stock = add_indicators(stock)
merged = merge_data(stock, news)              # uses DJIA Label directly
merged = merged.sort_values('Date').reset_index(drop=True)
merged = merged.ffill()

print(f"Merged rows: {len(merged)}")
print(f"Label distribution:\n{merged['Label'].value_counts()}")

# ── NORMALIZE ─────────────────────────────────────────────────
FEATURE_COLS = ['Open','High','Low','Close','Volume',
                'MA5','MA20','EMA','MACD','MACD_Signal','MACD_Hist',
                'RSI','Volatility','Momentum','BB_upper','BB_lower','BB_pct',
                'Return1','Return5','VolRatio']

merged = normalize_features(merged, FEATURE_COLS)

# ── TOKENIZE ──────────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
texts = merged['News'].tolist()

enc = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

# ── TENSORS ───────────────────────────────────────────────────
price_data = torch.tensor(merged[FEATURE_COLS].values, dtype=torch.float32)
labels     = torch.tensor(merged['Label'].values.astype(int), dtype=torch.long)

# ── SEQUENCES ─────────────────────────────────────────────────
SEQ_LEN = 10

price_seq, ids_seq, mask_seq, labels_seq = create_sequences(
    price_data, enc['input_ids'], enc['attention_mask'], labels, seq_len=SEQ_LEN
)

print(f"Sequences: {price_seq.shape[0]} | Price dim: {price_seq.shape[-1]}")

# ── SPLIT ─────────────────────────────────────────────────────
X_p_train, X_p_test, ids_train, ids_test, mask_train, mask_test, y_train, y_test = train_test_split(
    price_seq, ids_seq, mask_seq, labels_seq,
    test_size=0.2, random_state=42, stratify=labels_seq
)

print(f"Train: {len(X_p_train)} | Test: {len(X_p_test)}")
print(f"Train label dist: {torch.bincount(y_train)}")
print(f"Test  label dist: {torch.bincount(y_test)}")

# ── MODEL ─────────────────────────────────────────────────────
model = MultimodalModel(price_dim=price_seq.shape[-1], num_classes=2)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\nTrainable params: {trainable:,} / {total:,}")

# ── TRAIN ─────────────────────────────────────────────────────
best_acc = train_model(
    model,
    X_p_train, ids_train, mask_train, y_train,
    X_p_test,  ids_test,  mask_test,  y_test,
    epochs=20, patience=5
)

# ── EVALUATE ──────────────────────────────────────────────────
metrics = evaluate(model, X_p_test, ids_test, mask_test, y_test)
print("\n=== FINAL METRICS ===")
for k, v in metrics.items():
    print(f"  {k:12s}: {v:.4f}")
