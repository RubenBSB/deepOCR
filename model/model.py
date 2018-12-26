import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dSame(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ZeroPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )
    def forward(self, x):
        return self.net(x)

class DeepOCR(nn.Module):

    def __init__(self,input_size):
        super(DeepOCR,self).__init__()
        assert(isinstance(input_size,tuple))

        input_h, input_w = input_size

        self.conv1 = Conv2dSame(1,32,5)
        self.conv2 = Conv2dSame(32,64,5)
        self.conv3 = Conv2dSame(64,128,3)
        self.conv4 = Conv2dSame(128,128,3)
        self.conv5 = Conv2dSame(128,256,3)

        self.pool1 = nn.MaxPool2d((2,2))
        self.pool2 = nn.MaxPool2d((2,1))

        self.lstm = nn.LSTM(int(256*input_h/32),256,num_layers=2)

        self.fc = nn.Linear(256,79)


    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool2(F.relu(self.conv4(x)))
        x = self.pool2(F.relu(self.conv5(x)))
        x = x.view(x.size(0),x.size(1)*x.size(2),-1)
        x = x.permute((2,0,1))
        x,_ = self.lstm(x)
        x = self.fc(x)
        x = F.log_softmax(x,-1)

        return x