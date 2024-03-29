import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Path to the directory containing your dataset
data_dir = os.getcwd() + "/train/cleaned"

# List all emotion folders
emotions = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]


# Function to display a grid of 25 images
def display_images(emotion):
    print("---")
    plt.figure(figsize=(10, 10))
    plt.suptitle(emotion, fontsize=16)
    image_files = os.listdir(os.path.join(data_dir, emotion))
    selected_images = random.sample(image_files, 25)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        img = mpimg.imread(os.path.join(data_dir, emotion, selected_images[i]))
        print(selected_images[i])
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.savefig(os.path.join(os.getcwd(), f"{emotion}_grid.png"))  # Save the grid
    plt.show()
    plt.close()


# Display 25 random images for each emotion
for emotion in emotions:
    display_images(emotion)
