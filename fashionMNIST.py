# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:39:47 2019
This code follows the tutorial available by youtube channel 'deep lizard'
as means of learning the basic tools offered by pytorch libraries to implement 
and train Convolutional Neural Networks.
@author: CaioChaves
"""

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn # Pytorch neural network library
import torch.optim as optim
import argparse


# In the lines below, we declare a pytorch class that extends the
# pytorch nn.Module class (INHERITANCE!)
# Thus, the weights of each layer will be stored as learnable parameters
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        # We will define the layers of the network as class attributes
        # conv1 -> input channels equals one because it is a gray scale
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4,out_features=120,bias=True)
        self.fc2 = nn.Linear(in_features=120,out_features=60,bias=True)
        self.out = nn.Linear(in_features=60,out_features=10,bias=True)
        
    def forward(self,t):
        
        # t: tensor input
        
        # conv1 hidden layer
        t = self.conv1(t)
        t = F.relu(t)           # Activation function
        t = F.max_pool2d(t,kernel_size=2,stride=2)
        
        # conv2 hidden layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size=2,stride=2)
        
        # fc1 hidden layer
        t = t.reshape(-1,12*4*4)
        t = self.fc1(t)
        t = F.relu(t)
        
        # fc2 hidden layer
        t = self.fc2(t) 
        t = F.relu(t)
        
        # output layer
        t = self.out(t)
        t = F.softmax(t,dim=1) # Return the probabilities for each class (sum=1) 
        
        return t

"""
## FORWARD PROPAGATION!!
# input shape needs to be (batch size x input color channels x H x W)
prediction = myNetwork(image.unsqueeze(0)) 

prediction_batch = myNetwork(images)
"""

def labels2tensor(labels):
    t = torch.zeros(len(labels),10) # 10 classes
    for i in range(len(labels)):
        t[i][labels[i]] = 1;
    return t

def my_loss_function(prediction,labels):
    # This function computes the loss function as the norm L2 of the error
    return torch.sqrt(torch.sum(torch.pow(prediction-labels,2)))
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    lossEpoch = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        lossEpoch.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return np.mean(lossEpoch)

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_loss,(100. * correct / len(test_loader.dataset))


def main():
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    
    torch.set_printoptions(linewidth=120)

    # For the forward propagation step, it is not necessary to keep track of the 
    # gradients of the weights (they will be useful later in backprop). To safe 
    # memory, let's turn off the storage of these parameters
    # torch.set_grad_enabled(False)
    
    device = torch.device("cpu")
    
    # Extract the data - Get the Fashion-MNIST image data from the source
    # Transform the data - Put our data into tensor form
    train_set = torchvision.datasets.FashionMNIST(
        root = './data/FashionMNIST',
        train = True,
        download = True,
        transform = transforms.Compose([
                transforms.ToTensor()
                ])        
        )
    
    
    # Load the data - Put our data into an object to make it easily accessible
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=args.batch_size
                                               )
                   

    # Test set
    test_set = torchvision.datasets.FashionMNIST(
            root = './data/FashionMNIST',
            train = True,
            download = True, 
            transform = transforms.Compose([
                    transforms.ToTensor()
                    ])  
            )
                  
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.test_batch_size
                                              )
    
      
    # Get a sample from the dataset
    image,label = next(iter(train_set))
    """
    plt.imshow(image.squeeze(),cmap='gray')
    print('label:',label)
    """
    
    images,labels = next(iter(train_loader))
    
    myNetwork = Network() # Creation of instance
    optimizer = optim.SGD(myNetwork.parameters(),lr=args.lr,momentum=args.momentum)
    
    lossTrainPile = []
    lossTestPile = []
    accuracyPile = []
    
    """"
    for epoch in range(1, args.epochs + 1):
        lossEpochTrain = train(args, myNetwork, device, train_loader, optimizer, epoch)
        lossTrainPile.append(lossEpochTrain)
        lossEpochTest,accuracyEpoch = test(args, myNetwork, device, test_loader)
        lossTestPile.append(lossEpochTest)
        accuracyPile.append(accuracyEpoch)

    if (args.save_model):
        torch.save(myNetwork.state_dict(),"fasion_mnist_cnn.pt")

    visualize_plot = True
    
    if visualize_plot == True:
        visualize_learning_statistics(lossTrainPile,lossTestPile,accuracyPile)
    """


def visualize_learning_statistics(lossTrain,lossTest,accuracy):
    fig, ax = plt.subplots(1, 2)

    ax[0].plot(lossTrain, label='Train') #row=0, col=0
    ax[0].plot(lossTest, label='Test') #row=0, col=0
    plt.xlabel('Epochs')
    plt.ylabel('Loss function')
    plt.legend()
    
    ax[1].plot(accuracy) #row=0, col=1
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    
    
    plt.show()
    
    
if __name__ == '__main__':
    main()
