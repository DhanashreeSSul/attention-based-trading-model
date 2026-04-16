import torch
import torch.nn as nn
from transformers import BertModel


# TEXT ENCODER — partial fine-tuning (last 2 layers unfrozen)
class FinBERTEncoder(nn.Module):
    def __init__(self, unfreeze_layers=2):
        super().__init__()
        self.bert = BertModel.from_pretrained("yiyanghkust/finbert-tone")

        for param in self.bert.parameters():
            param.requires_grad = False

        total_layers = len(self.bert.encoder.layer)
        for i in range(total_layers - unfreeze_layers, total_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True
        for param in self.bert.pooler.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        mean_pool = outputs.last_hidden_state.mean(dim=1)
        return cls + mean_pool, outputs.last_hidden_state


# PRICE ENCODER — bidirectional LSTM
class PriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.out_dim = hidden_dim * 2

    def forward(self, x):
        out, (hn, _) = self.lstm(x)
        final = torch.cat([hn[-2], hn[-1]], dim=-1)
        return final, out


# CROSS ATTENTION
class CrossAttention(nn.Module):
    def __init__(self, price_dim=128, text_dim=768, num_heads=8):
        super().__init__()
        self.query_proj = nn.Linear(price_dim, text_dim)
        self.attn = nn.MultiheadAttention(embed_dim=text_dim, num_heads=num_heads,
                                          batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(text_dim)

    def forward(self, price_seq, text_seq):
        query = self.query_proj(price_seq)
        attn_out, _ = self.attn(query, text_seq, text_seq)
        attn_out = self.norm(attn_out + query)
        return attn_out.mean(dim=1)


# MAIN MODEL
class MultimodalModel(nn.Module):
    def __init__(self, price_dim, num_classes=2):
        super().__init__()

        self.text_encoder = FinBERTEncoder(unfreeze_layers=2)
        self.price_encoder = PriceLSTM(price_dim, hidden_dim=64)
        self.cross_attn = CrossAttention(price_dim=128, text_dim=768, num_heads=8)

        fused_dim = 768 + 768 + 128

        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, price_x, input_ids, attention_mask):
        text_cls, text_seq = self.text_encoder(input_ids, attention_mask)
        price_summary, price_seq = self.price_encoder(price_x)
        cross_feat = self.cross_attn(price_seq, text_seq)
        fused = torch.cat([cross_feat, text_cls, price_summary], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)
        return self.fc(fused)
