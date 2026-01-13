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
classes = data.classes
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
    plt.title("Example images from the training dataset")
    plt.show()
dataiter = iter(train_loader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
plt.savefig("example_training_images.png")

## Defining a CNN - Inspiried by class notes
print("\n\n>>> Defining a CNN")
class Net(nn.Module):
    def __init__(self, num_classes = 6, input_size=128):
        super().__init__()
        ## Conv layers
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        
        ## FN layers
        self.fc1 = nn.Linear(16*29*29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    def forward(self, x):
        ## conv layers, relu and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        ## Flattern
        x = torch.flatten(x, 1)
        ## FC layer, relu
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        ## Output later
        x = self.fc3(x)
        return x
##Create network
net = Net(num_classes=6)
print(net)

##Defining loss function and optimiser
## Adam optimiser - originally tried with SGD but not as efficient or accurate
print("\n\n>>> Defining Cross Entropy Loss and Adam optimiser (learning rate = 0.001)")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.to(device)

## Training the network
## Training for 15 epochs
print("\n\n>>> Training network, 15 batches")
for epoch in range(10):
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        ## Print statistics every 10 batches
        if (i+1) % 10 ==0:
            print(f"Epoch {epoch+1}, Batch {i+1} Loss: {running_loss / 10:.3f}")
            running_loss = 0.0
print("\n\n>>> Finished Training")

## Save the model
PATH = "./bird_classifier.pth"
torch.save(net.state_dict(), PATH)
print("\n\n>>> Saved model to ./bird_classifier.pth")


## Initialising correct prediction and total prediction dictionaries
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

## Testing model and saving predictions 
print(">>> Testing model")
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        
        for label, prediction in zip(labels, predictions):
            label = label.item()
            prediction = prediction.item()
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1
# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("\n\n>>> Testing completed, results follow")
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')