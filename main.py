import torch
import torch.nn as nn
import torch.optim as optim
import config
from src.data_loader import get_mnist_dataloader
from src.model import MNISTModel

def train():
    #hardware configuration
    print(f"Using device : {config.DEVICE}")

    #get data loaders
    train_loader, test_loader = get_mnist_dataloader(config.BATCH_SIZE)
    model = MNISTModel().to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        for batchidx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)

            #forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")
        evaluate(model, test_loader)
    #save the model
    torch.save(model.state_dict(), config.SAVE_PATH)
    print(f"Model saved to {config.SAVE_PATH}")

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
    
            scores = model(data)
            _, predictions = torch.max(scores, 1)

            total += targets.size(0)
            correct += (predictions == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    train()