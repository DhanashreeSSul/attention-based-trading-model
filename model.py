import torch
import torch.nn as nn
from transformers import BertModel

class FinBERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("yiyanghkust/finbert-tone")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

class PriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]


class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=2, batch_first=True)

    def forward(self, price_feat, text_feat):
        price_feat = price_feat.unsqueeze(1)
        text_feat = text_feat.unsqueeze(1)
        fused, _ = self.attn(price_feat, text_feat, text_feat)
        return fused.squeeze(1)

class MultimodalModel(nn.Module):
    def __init__(self, price_dim):
        super().__init__()

        self.text_encoder = FinBERTEncoder()
        self.price_encoder = PriceLSTM(price_dim)

        # 🔥 ADD THIS (IMPORTANT)
        self.price_projection = nn.Linear(64, 768)

        self.fusion = AttentionFusion(768)
        self.fc = nn.Linear(768, 2)

    def forward(self, price_x, input_ids, attention_mask):
        text_feat = self.text_encoder(input_ids, attention_mask)   # (batch, 768)
        price_feat = self.price_encoder(price_x)                    # (batch, 64)

        # 🔥 FIX: convert 64 → 768
        price_feat = self.price_projection(price_feat)

        fused = self.fusion(price_feat, text_feat)

        return self.fc(fused)