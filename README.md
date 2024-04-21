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
- 2 image_brightening.py: Script responsible for adjusting the brightness of the images so they have similar brightness for better image processing by the model.
## Visulization:
- 1 visualization_class_distribution.py:
  Script responsible for generating the bar graph showing the number of images in each class.
- 2 visulaization_sample_images.py:
  Script responsible for randomly selecting 25 images sample and ploting a grid for each class in the dataset.

# Initil Model Development:
## Data preprocessing: 
- 1 load_data.py:
  This file is used to split and load the data from the dataset folder and feed it to the training loop.
## Model defining: 
- 1 CNN_def.py:
contains different CNN architectures we implemented in the process of defining a model with high accuracy.
## Training and Testing: 
- 1 train.py:
  In this file we defined the loop that is responsible for training the models defined in CNN_def and generating **.pt** files.
- 2 test_on_dataset.py and test_on_an_image.py:
  These files are used to test the models using the test dataset or an specific image, where we feed the models the data and test their response against what we know regarding the data.
