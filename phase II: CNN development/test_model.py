# Test the best model
import numpy as np
import torch
from torch import nn
from Load_data import load_data
# from CNN_def import EMModel
from CNN_def import EmotionCNN4
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


model = EmotionCNN4()

# load and split images from the dataset
train_loader, val_loader, test_loader = load_data()

criterion = nn.CrossEntropyLoss()

model.load_state_dict(torch.load('best_model.pt'))
model.eval()

test_loss = 0.0
correct = 0
total = 0

# # Initialize variables for true positives, false positives, false negatives
# true_positives = 0
# false_positives = 0
# false_negatives = 0
#
# # Class-wise counts for precision and recall
# class_counts = torch.zeros(7)  # Assuming there are 7 classes, adjust as needed
#
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         outputs = model(inputs)
#         test_loss += criterion(outputs, labels).item()
#         _, predicted = torch.max(outputs, 1)
#
#         for i in range(len(labels)):
#             true_label = labels[i].item()
#             predicted_label = predicted[i].item()
#
#             if true_label == predicted_label:
#                 true_positives += 1
#                 class_counts[true_label] += 1
#             else:
#                 false_positives += 1
#                 false_negatives += 1
#                 class_counts[true_label] += 1
#                 class_counts[predicted_label] += 1
#
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# test_loss /= len(test_loader)
# accuracy = 100 * correct / total
#
# print(total)
# print(true_positives)
# print(false_negatives)
# print(false_positives)
#
# # Compute precision, recall, and F-measure
# precision = true_positives / (true_positives + false_positives)
# recall = true_positives / (true_positives + false_negatives)
# f_measure = 2 * (precision * recall) / (precision + recall)
#
# print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
# print(f'Precision: {precision:.4f}')
# print(f'Recall: {recall:.4f}')
# print(f'F-measure: {f_measure:.4f}')




# # Initialize lists to store predictions and ground truths
# predictions = []
# targets = []
#
# # Iterate through the dataset to make predictions
# for images, labels in test_loader:
#     with torch.no_grad():
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         predictions.extend(predicted.cpu().numpy())
#         targets.extend(labels.numpy())
#
# # Calculate metrics
# accuracy = accuracy_score(targets, predictions)
# precision = precision_score(targets, predictions, average='micro')
# recall = recall_score(targets, predictions, average='micro')
# f1 = f1_score(targets, predictions, average='micro')
#
# print("Accuracy: {:.4f}".format(accuracy))
# print("Precision: {:.4f}".format(precision))
# print("Recall: {:.4f}".format(recall))
# print("F1-score: {:.4f}".format(f1))




num_classes = 4
# Initialize variables to store counts
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
