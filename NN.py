import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Resize
from torchvision.io import decode_image
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.io as io
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torchvision import datasets
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3)
        
        self.pool1 = nn.AdaptiveMaxPool2d((5, 5))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*5*5, 3),
            nn.ReLU(),
            nn.Linear(3, 3)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.classifier(x)
        return x

if (__name__ == '__main__'):
    path1 = "train_data/tomato/"
    path2 = 'train_data/strawberry/'
    path3 = 'train_data/cherry/'

    """
    tomato = os.listdir(path1)
    strawberry = os.listdir(path2)
    cherry = os.listdir(path3)

    transforms = transforms.Compose([transforms.ToTensor(), Resize((300, 300))])

    data = Image.open(path1 + tomato[0])

    data = transforms(data)

    targ = np.array("Tomato")

    for fname in tomato[1:len(tomato)]:
        img = Image.open(path1 + fname)
        img = transforms(img)
        if (img.shape[0] != 3):
            continue
        data = torch.cat((data, img), -1)
        targ = np.append(targ, "Tomato")

    for fname in strawberry:
        img = Image.open(path2 + fname)
        img = transforms(img)
        if (img.shape[0] != 3):
            continue
        data = torch.cat((data, img), -1)
        targ = np.append(targ, "Strawberry")

    for fname in cherry:
        img = Image.open(path3 + fname)
        img = transforms(img)
        if (img.shape[0] != 3):
            continue
        data = torch.cat((data, img), -1)
        targ = np.append(targ, "Cherry")

    print(data.shape)

    """

    transforms = transforms.Compose([transforms.ToTensor(), Resize((300, 300))])

    train = datasets.ImageFolder("train_data/", transform = transforms)

    train_set = DataLoader(train, batch_size = 500, shuffle = True, num_workers = 0)

    images, labels = next(iter(train_set))

    

    print(type(train_set))

    lossHistory = []
    accuHistory = []

    net = Net()

    lossFn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)

    for t in range(1000):
        for data in train_set:
            x, targ = data
            optimizer.zero_grad()

            y = net.forward(x)

            loss = lossFn(y, targ)

            loss.backward()

            optimizer.step()

            accuracy = torch.mean((torch.argmax(y, dim = 1) == targ).float())
            lossHistory.append(loss.detach().item())
            accuHistory.append(accuracy.detach())

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(lossHistory, 'r'); plt.title("loss"); plt.xlabel("epochs")
    plt.subplot(1,2,2)
    plt.plot(accuHistory, 'b'); plt.title("accuracy")

    plt.show()

    
