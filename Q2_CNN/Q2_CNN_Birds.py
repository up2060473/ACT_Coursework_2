data_path = "../../../Data_birds/birds_dataset/Bird Speciees Dataset/"
## Currently, this is accessing local file on sciserver
## To be updated to take dataset from Kaggle

## Importing Libraries 
## To be updated to be store in a dependancies folder
## Torch
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
## Standard libraries, such as numpy and matplot lib
import matplotlib.pyplot as plt
import numpy as np

## Which device?
print("##### Starting #####")
print("GPU Available: ", torch.cuda.is_available())

## Setting Device
if torch.cuda.is_available() == False:
    print("Switching to CPU.... \nMay result in slower performance")
    device = "cpu"
else:
    print("Cuda GPU found, switching device to increase performance")
    device = "cuda"