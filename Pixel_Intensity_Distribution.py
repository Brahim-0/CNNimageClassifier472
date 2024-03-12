import cv2
import os
import matplotlib.pyplot as plt

def plot_intensity_distribution(image_folder):
    # Initialize lists to store pixel intensities
    intensities_gray = []

    # Iterate through images in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read greyscaled image
            image_path = os.path.join(image_folder, filename)
            gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Calculate histogram
            hist_gray = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            intensities_gray.extend(hist_gray)

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.title('Pixel Intensity Distribution')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.plot(intensities_gray, color='black', label='Gray')
    plt.legend()
    plt.show()

# current path
current_path = os.getcwd()
print(current_path)

# Change this to the path of your image folder
image_folder = current_path + "/Archive/Focused_25_sample"
plot_intensity_distribution(image_folder)

image_folder = current_path + "/Archive/Happy_25_sample"
plot_intensity_distribution(image_folder)

image_folder = current_path + "/Archive/Neutral_25_sample"
plot_intensity_distribution(image_folder)

image_folder = current_path + "/Archive/Surprised_25_sample"

plot_intensity_distribution(image_folder)
