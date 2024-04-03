import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from CNN_def import EmotionCNN7, EmotionCNN7K, EmotionCNN7L
from load_data import load_data
import torch

# Load the test data
train_loader, val_loader, test_loader = load_data()

# Load the models
model_7 = EmotionCNN7()
model_7K = EmotionCNN7K()
model_7L = EmotionCNN7L()

model_7.load_state_dict(torch.load('CNN_development/best_model_E7_70.pt'))
model_7K.load_state_dict(torch.load('CNN_development/best_model_E7K_69.pt'))
model_7L.load_state_dict(torch.load('CNN_development/best_model_E7L_64.pt'))



def plot_confusion_matrix(model, test_loader, model_name):
    # Make predictions
    model.eval()
    all_predictions = []
    all_labels = []
 
    for images, labels in test_loader:
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    #Plot the matrix
    classes = ['happy', 'neutral', 'focused', 'surprised' ]
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    #saves to directory
    plt.savefig(f'CNN_development/confusion_matrix_{model_name}.png')
    #uncomment the line below if you want to see the plot
    #plt.show()
    print(cm)


# Generate confusion matrices for each model
plot_confusion_matrix(model_7, test_loader, "EmotionCNN7")
plot_confusion_matrix(model_7K, test_loader, "EmotionCNN7K")
plot_confusion_matrix(model_7L, test_loader, "EmotionCNN7L")
