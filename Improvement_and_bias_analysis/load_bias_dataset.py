from matplotlib import pyplot as plt
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torch
from torchvision.utils import make_grid

def load_bias_dataset():
    # define the dir for the dataset
    data_dir = './dataset'
    dataset = ImageFolder(data_dir, transform=ToTensor())

    # Create data loaders for training and testing
    dataLoader = DataLoader(dataset, batch_size=32, shuffle=False)

    return dataLoader