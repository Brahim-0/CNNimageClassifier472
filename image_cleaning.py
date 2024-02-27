import os
import cv2


def check_image_sizes(directory):
    images_per_dimension = {}

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # Check if the file is an image
        if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # Get image dimensions
            height, width = image.shape[:2]

            # Update the dictionary
            if (height, width) not in images_per_dimension:
                images_per_dimension[(height, width)] = 1
            else:
                images_per_dimension[(height, width)] += 1

    return images_per_dimension


if __name__ == '__main__':
    # Directories containing the images
    folders = ['/train/Happy_Cleaned', '/train/surprised_Cleaned']

    for folder in folders:
        # Call the function to count images per dimension
        images_per_dimension = check_image_sizes(os.getcwd() + folder)
        print("result for" + "'" + folder + "':")
        # Print the results
        for dimension, count in images_per_dimension.items():
            print(f"Dimension: {dimension}, Number of Images: {count}")

