from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict



class Net(nn.Module):
    def __init__(self,norm_type='BN',dropout_value = 0.01):
        super(Net, self).__init__()
        self.conv1 = self.conv2d(1, 8, 3,norm_type,dropout_value,2)
        self.conv2 = self.conv2d(8, 16, 3,norm_type,dropout_value,4) 
        
        #Transition Block
        self.trans1 = nn.Sequential(
            
            nn.MaxPool2d(2, 2), #  Input 24x24 output 12x12 RF : 6x6
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False)  # Input 12x12 output 12x12 RF : 6x6
        )
        
        self.conv3 = self.conv2d(8, 16, 3,norm_type,dropout_value,4) 
        self.conv4 = self.conv2d(16, 16, 3,norm_type,dropout_value,4) 
        self.conv5 = self.conv2d(16, 16, 3,norm_type,dropout_value,4) 
        self.avgpool2d = nn.AvgPool2d(kernel_size=6)
        self.conv6 =  self.conv2d(16, 16, 1,norm_type,dropout_value,4) 
        self.conv7 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False) 

    def conv2d(self, in_channels, out_channels, kernel_size, norm_type, dropout,num_of_groups):
        if norm_type == "BN":
         conv = nn.Sequential(OrderedDict([
                ('conv2d', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding=0, bias=False)),
                ('Relu', nn.ReLU()),
                ('BatchNorm',nn.BatchNorm2d(out_channels)),
                ('Dropout', nn.Dropout(dropout))
         ]))
        elif norm_type == "LN":
            conv = nn.Sequential(OrderedDict([
                ('conv2d',nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0, bias=False)),
                ('Relu', nn.ReLU()),
                ## When number of groups is 1, its layernorm
                ('LayerNorm',nn.GroupNorm(1,out_channels)),
                ('Dropout',nn.Dropout(dropout))
            ]))
        elif norm_type == "GN":
            conv = nn.Sequential(OrderedDict([
                ('conv2d',nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0, bias=False)),
                ('Relu', nn.ReLU()),
                ('GroupNorm',nn.GroupNorm(num_of_groups,out_channels)),
                ('Dropout',nn.Dropout(dropout))
            ]))
        else:
            conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0, bias=False),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
      
        return conv

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool2d(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
