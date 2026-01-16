import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

## Setting Device to GPU or CPU
def set_device():
    ##PyTorch (The python library we will be using to build and run the CNN) can run on either your machine's CPU or CUDA (Nvidia's parallel computing platform). CUDA generally provides a faster runtime for CNN models. 
    ##The code below searches to see if CUDA is available and switches to it if possible. 
    if torch.cuda.is_available() == False:
        print("Switching to CPU.... \nMay result in slower performance")
        device = "cpu"
    else:
        print("Cuda GPU found, switching device to increase performance")
        device = "cuda"
        
def imshow(img):
    ## Helper function to load image
    ## img: Reverses the normalisation applied in the transform 
    ## npimg: Covnerts PyTorch tensor to np.array - readabl by matplotlib
    ## np.transpose: Ensures that the height, width, channels are in correct order
    ## plt.imshow(...): Load image grid
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.axis("off")
    plt.title("Example images from the training dataset")
    plt.show()
        
