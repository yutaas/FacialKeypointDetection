## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import numpy as np

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        self.pool = nn.MaxPool2d(3,3) # max pooling layer, kernel size = 3, stride = 3
        self.pool_2 = nn.MaxPool2d(2,2) # max pooling layer, kernel size = 3, stride = 3
        self.poolAve = nn.AvgPool2d(7,7)
        
        self.dropout = nn.Dropout(p=0.4)
        
        # 128 outputs * the 7*7 filtered/pooled map size
        # 136 output channels (for the 136 classes)
        self.fc1 = nn.Linear(256*2*2, 256)
        self.fc2 = nn.Linear(256, 136)
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = np.transpose(x, (0,3,1,2)) # dimensions: number of images, color channel, height, width
#         print(x.shape)
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
#         print(x.shape)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
#         print(x.shape)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
#         print(x.shape)
        
        
        x = F.relu(self.conv4(x))
        x = self.pool_2(x)
#         print(x.shape)
        
        
#         x = self.poolAve(x)
#         print(x.shape)
        
        x = x.view(x.size(0), -1) # flatten the inputs into a vector
        x = self.dropout(x)
#         print(x.shape)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
#         print(x.shape)
        
        x = F.relu(self.fc2(x))
#         print(x.shape)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
