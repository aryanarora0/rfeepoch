import numpy as np

def train_epoch(model, train_loader, criterion, optimizer, loss_log, n_epoch=10):
    model.train()
    for epoch in range(n_epoch):
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target.unsqueeze(1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_log.append(avg_loss)

        print(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')

    return avg_loss