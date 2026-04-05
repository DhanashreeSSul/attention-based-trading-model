import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score


def train_model(model,
                X_price_train, X_text_ids_train, X_mask_train, y_train,
                X_price_test, X_text_ids_test, X_mask_test, y_test,
                epochs=14):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model.to(device)

    train_dataset = TensorDataset(X_price_train, X_text_ids_train, X_mask_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        train_preds = []
        train_labels = []

        print(f"\nEpoch {epoch+1} starting...")

        # 🔹 TRAINING LOOP
        model.train()
        for i, (price, ids, mask, label) in enumerate(train_loader):
            price = price.to(device)
            ids = ids.to(device)
            mask = mask.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = model(price, ids, mask)
            loss = loss_fn(output, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(output, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(label.cpu().numpy())

            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item():.4f}")

        # ✅ TRAIN ACCURACY
        train_acc = accuracy_score(train_labels, train_preds)

        # 🔹 TEST LOOP
        model.eval()
        test_preds = []

        with torch.no_grad():
            for i in range(len(y_test)):
                out = model(
                    X_price_test[i].unsqueeze(0).to(device),
                    X_text_ids_test[i].unsqueeze(0).to(device),
                    X_mask_test[i].unsqueeze(0).to(device)
                )
                pred = torch.argmax(out, dim=1).item()
                test_preds.append(pred)

        test_acc = accuracy_score(y_test.cpu().numpy(), test_preds)

        print(f"Epoch {epoch+1} completed | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")


    torch.save(model.state_dict(), "model.pth")
    print("Model saved!")