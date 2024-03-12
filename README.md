# CNNimageClassifier472
The objective of this project is to develop a Deep Learning Convolutional Neural Network (CNN)
using PyTorch that can analyze images of students in a classroom or online meeting setting and
categorize them into distinct states or activities. Our system will be able to analyse images in
order to recognize four classes based on emotions.

# Dataset:
we used a relativley clean dataset available via the link below. the dataset provide a large collection of grey scaled image of the size 25x25.
https://www.kaggle.com/datasets/vipulpisal/fer2013-updated

# Scripts:
## Cleaning: 
- 1 image_cleaning.py:
  Running this script will go through the data and make sure the image have the same size. the script will return an analysis if the images and their size in the dataset.

## Visulization:
- 1 visualization_class_distribution.py:
  Script responsible for generating the bar graph showing the number of images in each class.
- 2 visulaization_sample_images.py:
  Script responsible for randomly selecting 25 images sample and ploting a grid for each class in the dataset.
