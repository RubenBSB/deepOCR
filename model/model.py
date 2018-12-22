import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepOCR(nn.Module):

    def __init__(self,input_size):
        super(DeepOCR,self).__init__()

        input_h, input_w = input_size

        self.conv1 = nn.Conv2d(1,32,5)
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,3)
        self.conv4 = nn.Conv2d(128,128,3)
        self.conv5 = nn.Conv2d(128,256,3)

        self.pool1 = nn.MaxPool2d((2,2))
        self.pool2 = nn.MaxPool2d((2,1))

        self.lstm = nn.LSTM(256*input_h/32,256,num_layers=2)

        self.fc = nn.Linear(256,80)


    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool2(F.relu(self.conv4(x)))
        x = self.pool2(F.relu(self.conv5(x)))
        x = x.view(x.size(0),x.size(1)*x.size(2),-1)
        x = self.lstm(x)
        x = F.softmax(self.fc(x))

        return x