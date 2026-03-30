import torch
import numpy as np
from utils import calculate_metrics


def evaluate(model, X_price, X_text_ids, X_mask, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(len(y)):
            out = model(X_price[i].unsqueeze(0).to(device),
                        X_text_ids[i].unsqueeze(0).to(device),
                        X_mask[i].unsqueeze(0).to(device))
            preds.append(torch.argmax(out, dim=1).item())

    return calculate_metrics(y.numpy(), np.array(preds))
