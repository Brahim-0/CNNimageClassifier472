import os
import matplotlib.pyplot as plt

# Path to the directory containing your dataset
data_dir = os.getcwd() + "/train/cleaned"

# List all emotion folders
emotions = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

# Count the number of images in each class
class_counts = {}
for emotion in emotions:
    class_counts[emotion] = len(os.listdir(os.path.join(data_dir, emotion)))

# Plotting the bar graph
plt.figure(figsize=(10, 6))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.xlabel('Emotions')
plt.ylabel('Number of Images')
plt.title('Class Distribution of Emotions')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()







