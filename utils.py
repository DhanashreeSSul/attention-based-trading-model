import numpy as np
import torch


def create_sequences(price_data, text_ids, masks, labels, seq_len=5):
    X_price, X_ids, X_mask, y = [], [], [], []

    for i in range(len(price_data) - seq_len):
        X_price.append(price_data[i:i+seq_len])
        X_ids.append(text_ids[i+seq_len])
        X_mask.append(masks[i+seq_len])
        y.append(labels[i+seq_len])

    return (
        torch.stack(X_price),
        torch.stack(X_ids),
        torch.stack(X_mask),
        torch.stack(y)
    )


def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=1),
        "recall": recall_score(y_true, y_pred, average='macro'),
        "f1": f1_score(y_true, y_pred, average='macro')
    }


def print_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def sharpe_ratio(returns):
    return np.mean(returns) / (np.std(returns) + 1e-8)