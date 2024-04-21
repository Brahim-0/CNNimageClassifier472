import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from CNN_def import EmotionCNN7
from load_data import load_data
import torch

# Load the test data
# test_loader, trai = load_data()

# # Load the models
# model = EmotionCNN7()

# model.load_state_dict(torch.load('best_model_kfold_81.pt'))

def plot_confusion_matrix(model, test_loader, model_name, num_classes):
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
    cm = confusion_matrix(all_labels, all_predictions, labels=np.arange(num_classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    #saves to directory
    plt.savefig(f'confusion_matrix_{model_name}.png')
    #uncomment the line below if you want to see the plot
    #plt.show()
    print(cm)


# Generate confusion matrices for each model
#plot_confusion_matrix(model, test_loader, "EmotionCNN7", 4)
