import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepOCR(nn.module):

    def __init__(self):
        super(DeepOCR,self).__init__()

        self.conv1 = nn.Conv2d(1,16,5)
        self.conv2 = nn.Conv2d(16,32,3)

        self.fc1 = nn.Linear(32,None)
        self.fc2 = nn.Linear(None, 30)

    def forward(self,x):