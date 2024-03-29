import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from load_data import load_data
from CNN_def import EmotionCNN4

# Initialize model, loss function, and optimizer
model = EmotionCNN4()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# load and split images from the dataset
train_loader, val_loader, test_loader = load_data()

# Training loop
best_val_loss = float('inf')
for epoch in range(35):  # 10 epochs
    model.train()
    train_loss = 0.0  # Initialize train loss

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()  # Accumulate train loss
    train_loss /= len(train_loader)  # Calculate average train loss

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')

    # check for early stopping
    if val_loss > 1.2 * best_val_loss:
        print("\ntraining was halt as the validation loss is increasing")
        break

