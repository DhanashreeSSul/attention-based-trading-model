import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import numpy as np


def train_model(model,
                X_price_train, X_text_ids_train, X_mask_train, y_train,
                X_price_test, X_text_ids_test, X_mask_test, y_test,
                epochs=15, patience=4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    train_dataset = TensorDataset(X_price_train, X_text_ids_train, X_mask_train, y_train)
    train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=False)

    # Class-weighted loss to handle imbalance
    class_counts = torch.bincount(y_train)
    class_weights = (1.0 / class_counts.float()).to(device)
    class_weights = class_weights / class_weights.sum()
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Separate LR for BERT layers vs. rest
    bert_params  = list(model.text_encoder.bert.parameters())
    other_params = [p for p in model.parameters() if not any(p is bp for bp in bert_params)]

    optimizer = torch.optim.AdamW([
        {'params': bert_params,  'lr': 2e-5, 'weight_decay': 0.01},
        {'params': other_params, 'lr': 5e-4, 'weight_decay': 1e-4},
    ])

    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[2e-5, 5e-4],
        total_steps=total_steps, pct_start=0.1
    )

    best_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_preds, train_labels = [], []

        for price, ids, mask, label in train_loader:
            price, ids, mask, label = price.to(device), ids.to(device), mask.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(price, ids, mask)
            loss = loss_fn(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(label.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)

        # Evaluation
        model.eval()
        test_preds = []
        test_dataset = TensorDataset(X_price_test, X_text_ids_test, X_mask_test, y_test)
        test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

        with torch.no_grad():
            for price, ids, mask, label in test_loader:
                price, ids, mask = price.to(device), ids.to(device), mask.to(device)
                outputs = model(price, ids, mask)
                preds = torch.argmax(outputs, dim=1)
                test_preds.extend(preds.cpu().numpy())

        test_acc = accuracy_score(y_test.cpu().numpy(), test_preds)
        print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best test acc: {best_acc:.4f}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), "model.pth")
    print(f"\nBest Test Accuracy: {best_acc:.4f}")
    print("Model saved!")
    return best_acc
