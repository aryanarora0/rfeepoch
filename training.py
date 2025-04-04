import numpy as np

def train_epoch(model, train_loader, criterion, optimizer, device, loss_log, epoch=10):
    model.train()
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

    return avg_loss