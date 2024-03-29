import numpy as np
import torch
from torch import nn
from load_data import load_data
# from CNN_def import EMModel
from CNN_def import EmotionCNN4


model = EmotionCNN4()

# load and split images from the dataset
train_loader, val_loader, test_loader = load_data()

criterion = nn.CrossEntropyLoss()

model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# Initialize variables to store counts
test_loss = 0.0
correct = 0
total = 0
num_classes = 4
total_samples = 0
correct_predictions = 0
true_positives = np.zeros(num_classes)
false_positives = np.zeros(num_classes)
false_negatives = np.zeros(num_classes)

# Iterate through the dataset to make predictions
for images, labels in test_loader:
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
