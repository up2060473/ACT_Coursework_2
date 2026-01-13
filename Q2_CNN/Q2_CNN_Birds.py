data_path = "../../../Data_birds/birds_dataset/Bird Speciees Dataset/"
## Currently, this is accessing local file on sciserver
## To be updated to take dataset from Kaggle
print("\n>>> Starting ")
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
print("\n>>> Device\nGPU Available: ", torch.cuda.is_available())

## Setting Device
if torch.cuda.is_available() == False:
    print("Switching to CPU.... \nMay result in slower performance\n\n")
    device = "cpu"
else:
    print("Cuda GPU found, switching device to increase performance\n\n")
    device = "cuda"

## Transforms
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])    

## Loading dataset and defining batch size
data = torchvision.datasets.ImageFolder(
    root = data_path,
    transform = transform,
)
batch_size = 16

## Investigating the distribution of samples per class in the dataset
#print(dict(Counter(data.targets)))
#print(data.class_to_idx)

#counter = dict(Counter(data.targets))
#index = dict(data.class_to_idx)
#counts = dict({tuple(index.keys()):tuple(counter.values())})
#print(counts)

#class_counts = Counter(label for _, label in data)
#for class_idx, count in class_counts.items():
#    class_name = data.classes[class_idx]
#    print(class_name, " : ", count, " samples")

## Investigation distribution
print(">>> Investigating distribution")
class_counts = Counter(data.targets)
for index, count in class_counts.items():
    print(f"{data.classes[index]} : {count}")
    
## Splitting data
## train size = 80%
print("\n\n>>> Splitting data into 80% training and 20% testing")
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = random_split(data, [train_size, test_size])

## Data_loaders
## num_workers = 0 as 2 returned error - sciserver
train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 0
)

test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = 0
)
## saving figures to .txt. Originally from class notes, but updated to save in a .txt file
print("\n\n>>> Saving a sample of images to images.png")
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.axis("off")
    plt.title("Example Images")
    plt.show()
dataiter = iter(train_loader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
plt.savefig("images.png")

