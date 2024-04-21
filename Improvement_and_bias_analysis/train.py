import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from load_data import load_data
from CNN_def import EmotionCNN7
from sklearn.model_selection import KFold

# Initialize model, loss function, and optimizer
model = EmotionCNN7()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load data
_, train_dataset = load_data()

# K-fold Cross-validation
k_folds = 10
kfold = KFold(n_splits=k_folds, shuffle=True)
best_val_loss = float('inf')

for fold, (train_indices, val_indices) in enumerate(kfold.split(train_dataset)):
    print(f'Fold {fold + 1}')

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

    # Training loop
    epochs = 4
    for epoch in range(epochs):
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

        # Validation epoch
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

    # Validation fold
    model.eval()

    # Initialize variables to store counts
    val_loss = 0.0
    correct = 0
    total = 0
    num_classes = 4
    total_samples = 0
    correct_predictions = 0
    true_positives = np.zeros(num_classes)
    false_positives = np.zeros(num_classes)
    false_negatives = np.zeros(num_classes)

    # Iterate through the dataset to make predictions
    for images, labels in val_loader:
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Count total samples and correct predictions for accuracy
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Count true positives, false positives, and false negatives for each class
            for i in range(num_classes):
                true_positives[i] += ((predicted == i) & (labels == i)).sum().item()
                false_positives[i] += ((predicted == i) & (labels != i)).sum().item()
                false_negatives[i] += ((predicted != i) & (labels == i)).sum().item()

    # Calculate accuracy
    accuracy = correct_predictions / total_samples

    # Calculate precision, recall, and F1-score for each class
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    # Calculate macro-averaged precision, recall, and F1-score
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # Calculate micro-averaged precision, recall, and F1-score
    micro_precision = np.sum(true_positives) / np.sum(true_positives + false_positives)
    micro_recall = np.sum(true_positives) / np.sum(true_positives + false_negatives)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

    print("Accuracy: {:.4f}".format(accuracy))
    print("Macro Precision: {:.4f}".format(macro_precision))
    print("Macro Recall: {:.4f}".format(macro_recall))
    print("Macro F1-score: {:.4f}".format(macro_f1))
    print("Micro Precision: {:.4f}".format(micro_precision))
    print("Micro Recall: {:.4f}".format(micro_recall))
    print("Micro F1-score: {:.4f}".format(micro_f1))

    # # check for early stopping (if the validation loss has increased by 30% compared to the lowest val loss value recorded before)
    # if val_loss > 1.3 * best_val_loss:
    #     print("\nTraining halted as the validation loss is increasing")
    #     break
