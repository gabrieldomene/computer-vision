import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.module):
    
    def __init__(self):
        super(Net, self).__init__()
        # This network takes a square grayscale image as input
        # 1 input channel, 32 output channels, 5x5 conv kernel
        # Ends with linear layer (keypoints), 135 values for (x, y) positions
        self.conv1 = nn.Conv2d(1, 32, 5)
        # output = (w - f)/s + 1 = (32 -5)/1 + 1 = 28
        # (32, 28, 28)
        # after pool layer
        # (32, 14, 14)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Second conv 14 input, 10 output 3x3 conv
        # output = (14 - 3)/1 + 1 = 6
        # (10, 6, 6)
        # after pooling
        # (10, 3, 3)
        self.conv2 = nn.Conv2d(14, 10, 3)
        
        # 10 outputs * 3*3 filters
        self.fc1 = (10*3*3, 50)
        self.fc1_drop = nn.Dropout(p=0.4)
        
        self.fc2 = (50, 136)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        return x