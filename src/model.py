import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()

        #Layer 1: Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,kernel_size=3, padding=1)

        #Layer2: Convolutional Layer
        self.conv2v = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        #layer 3: pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Fully connected layer 1
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

        #Fully connected layer 2 (Output layer)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2v(x)))

        x = x.view(-1, 64 * 7 * 7)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x