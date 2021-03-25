import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNetwork(nn.Module):
    """
    This class defines a simple and minimalistic CNN model
    The goal is to perform binary classification in a patch taken from
    a road scenes dataset (KittiDataset, CittyScapes, etc.)
    Input: RGB patch with shape [batch_size,3,28,28]
    Output: Scalar [probability of being navigable ground, between 0 and 1]
                    
    """ 
    def __init__(self,in_channels=3,num_classes_out=1):
        super(MyNetwork,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=32,kernel_size=[3,3],bias=True)
        self.pool = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(3,3)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=[3,3],bias=True)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=[3,3],bias=True)
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,10)
        self.fc3 = nn.Linear(10,num_classes_out)
        self.out = nn.Sigmoid()
        
    def forward(self,x):
        """
        Forward step: Input x has (shape numbers_images x channels x pixels_y x pixels_x)
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = x.view(-1,128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.out(x)

class MultiScalePatchNet(nn.Module):
    def __init__(self,in_channels=3,num_classes_out=1,spatial_features=None):
        super(MultiScalePatchNet,self).__init__()

        self.conv1_s = nn.Conv2d(in_channels=in_channels,out_channels=32,kernel_size=[3,3],bias=True)
        self.conv2_s = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=[3,3],bias=True)

        self.conv1_m = nn.Conv2d(in_channels=in_channels,out_channels=32,kernel_size=[3,3],bias=True)
        self.conv2_m = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=[3,3],bias=True)
        self.conv3_m = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=[3,3],bias=True)

        self.conv1_L = nn.Conv2d(in_channels=in_channels,out_channels=32,kernel_size=[3,3],bias=True)
        self.conv2_L = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=[3,3],bias=True)
        self.conv3_L = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=[3,3],bias=True)

        self.pool1_s = nn.MaxPool2d(kernel_size=2,stride=2)
        self.pool2_s = nn.MaxPool2d(kernel_size=4,stride=4)

        self.pool1_m = nn.MaxPool2d(kernel_size=2,stride=2)
        self.pool2_m = nn.MaxPool2d(kernel_size=3,stride=3)

        self.pool1_L = nn.MaxPool2d(kernel_size=3,stride=3)
        self.pool2_L = nn.MaxPool2d(kernel_size=3,stride=2)
        self.pool3_L = nn.MaxPool2d(kernel_size=3,stride=3)
        
        self.MS1 = nn.Conv1d(in_channels=320, out_channels=128, kernel_size = 1)            
        self.MS2 = nn.Conv1d(in_channels=128 ,out_channels=64, kernel_size = 1)        

        self.spatial_features = spatial_features
        if self.spatial_features:
            self.MS3 = nn.Conv1d(in_channels=66 ,out_channels=10, kernel_size = 1)
        else:
            self.MS3 = nn.Conv1d(in_channels=64 ,out_channels=10, kernel_size = 1)
        
        self.MS4 = nn.Conv1d(in_channels=10 ,out_channels=num_classes_out, kernel_size = 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self,small_patch,medium_patch,large_patch,device,pho=None,theta=None):
        ## Small
        x_s = F.relu(self.conv1_s(small_patch))
        x_s = self.pool1_s(x_s)
        x_s = F.relu(self.conv2_s(x_s))
        x_s = self.pool2_s(x_s)
        ## Medium
        x_m = F.relu(self.conv1_m(medium_patch))
        x_m = self.pool1_m(x_m)
        x_m = F.relu(self.conv2_m(x_m))
        x_m = self.pool1_m(x_m)
        x_m = F.relu(self.conv3_m(x_m))
        x_m = self.pool2_m(x_m)
        ## Large
        x_L = self.pool1_m(large_patch)
        x_L = F.relu(self.conv1_L(x_L))
        x_L = self.pool1_m(x_L)
        x_L = F.relu(self.conv2_L(x_L))
        x_L = self.pool1_m(x_L)
        x_L = F.relu(self.conv3_L(x_L))
        x_L = self.pool2_m(x_L)
        ## Concatenate
        x = torch.cat((x_s,x_m,x_L),dim=1)
        x = F.relu(self.MS1(x.view(-1,320,1)))
        x = F.relu(self.MS2(x))

        if self.spatial_features:
            x = torch.cat((x,torch.tensor([pho,theta],device=device).view(-1,2,1)),dim=1)   ##Including spatial features        
        
        x = F.relu(self.MS3(x))
        x = self.sigmoid(self.MS4(x))
        return x

class MinimalisticCNN(nn.Module):
    def __init__(self,in_channels=3,num_classes_out=1,spatial_features=None):
        super(MinimalisticCNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=15,kernel_size=[4,4],stride=4,bias=True)
        self.conv2 = nn.Conv2d(in_channels=15,out_channels=30,kernel_size=[3,3],bias=True)

        if spatial_features: 
            self.conv3 = nn.Conv1d(in_channels=32 ,out_channels=1, kernel_size = 1)
        else:   
            self.conv3 = nn.Conv1d(in_channels=30 ,out_channels=1, kernel_size = 1)

        self.pool = nn.MaxPool2d(kernel_size=3,stride=2)
        self.sigmoid = nn.Sigmoid()
        self.spatial_features = spatial_features


    def forward(self,medium_patch,device,pho=None,theta=None):
        x = F.relu(self.conv1(medium_patch))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = x.view(-1,30,1)
        if self.spatial_features:
            x = torch.cat((x,torch.tensor([pho,theta],device=device).view(-1,2,1)),dim=1)   ##Including spatial features        
        x = self.conv3(x)
        return self.sigmoid(x)


