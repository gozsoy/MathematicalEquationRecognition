import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Net1(nn.Module): # shallow CNN
    def __init__(self,total_class_size):
        super(Net1,self).__init__()
        self.conv1=nn.Conv2d(3,160,kernel_size=5)
        self.conv2=nn.Conv2d(160,320,kernel_size=5)
        self.max=nn.MaxPool2d(3)
        self.fcn=nn.Linear(15680,total_class_size)

    def forward(self, x):
        #x=torch.unsqueeze(x,1)
        x=F.relu(self.max(self.conv1(x)))
        x=F.relu(self.max(self.conv2(x)))
        x=x.view(x.size(0),-1)
        x=self.fcn(x)
        return x


class Net2(nn.Module): # just fully connected layers, most primitive
    def __init__(self,total_class_size):
        super(Net2,self).__init__()
        self.fcn1=nn.Linear(64,32)
        self.fcn2=nn.Linear(32,16)
        self.fcn3=nn.Linear(16, total_class_size)

    def forward(self, x):
        #x=x.view(x.size(0),-1)
        x=self.fcn1(x)
        x=self.fcn2(x)
        x=self.fcn3(x)
        return x


class Net3(nn.Module): # deeper CNN
    def __init__(self,total_class_size):
        super(Net3,self).__init__()
        vgg=torchvision.models.vgg11_bn(True)
        for param in vgg.parameters():
            param.requires_grad = False
        num_ftrs = vgg.classifier[6].in_features
        vgg.classifier[6] = nn.Linear(num_ftrs, total_class_size)


    def forward(self, x):
        x=self.vgg(x)

        return x
