from matplotlib import pyplot as plt
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torch
from torchvision.utils import make_grid

def load_data():
    # define the dir for the dataset
    data_dir = './dataset'
    dataset = ImageFolder(data_dir, transform=ToTensor())

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define sizes for train and test splits
    train_size = int(0.8 * len(dataset))  # 80% train
    test_size = len(dataset) - train_size  # 20% test

    # Split dataset into train and test
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders for training and testing
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return test_loader, train_dataset

# define the dir for the dataset
data_dir = './dataset'
dataset = ImageFolder(data_dir, transform=ToTensor())
def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        plt.show()  # Keep the figure displayed
        break  # Only show the first batch\

