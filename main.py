#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:43:41 2019

@author: CaioChaves
"""

import torch
import torchvision
import argparse
from kitti_semantics_dataloader import KittiDataset
import torch.optim as optim
import torch.nn as nn # Pytorch neural network library
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import numpy as np
import time
import copy
from torch.optim import lr_scheduler

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    lossEpoch = []
    for batch_idx, (image, segmentation) in enumerate(train_loader):
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


def imshow(inp,title=None):
    #Imshow for tensor
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std*inp+mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    plt.show()
    if title is not None:
        plt.title(title)
    


def train_model(model,criterion,optimizer,scheduler,num_epochs,train_dataloader,test_dataloader,device):
    since = time.time() # Starts count
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch,num_epochs-1))
        print("-"*10)

        # (1/2) Training
        scheduler.step()
        model.train() # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        # Iterate over data.
        for image,segmentation in train_dataloader:
            image = image.to(device)
            segmentation = segmentation.to(device)  # Ground truth
            optimizer.zero_grad() # Zero the parameter gradients
            # Forward
            torch.set_grad_enabled(True)
            prediction = model(image)
            loss = criterion(prediction,segmentation)
            # Backward
            loss.backward()
            optimizer.step()
            # Statistics
            running_loss += loss.item()*image.size(0)
            
        # (2/2) Validation
        model.eval()
        running_loss = 0.0
        running_corrects = 0

    return 





def main():
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch KITTI Semantic Segmentation')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                        help='input batch size for testing (default: 20)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--plot-samples', default=False,
                        help='For displaying some images from the batch (<10)')
    args = parser.parse_args()
    
    ## END arguments
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    train_dataset = KittiDataset(os.getcwd(),
                        split = 'train',
                        target_type = 'semantic_binary_ground', 
                        transform = transforms.Compose([
                                     transforms.RandomCrop([370,1220]),
                                     transforms.ToTensor(),
                                     #transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
                                     ]))  

    

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=args.batch_size
                                                 )
    

    test_dataset = KittiDataset(os.getcwd(),
                        split = 'test',
                        transform = transforms.Compose([
                                    transforms.RandomCrop([370,1220]),
                                    transforms.ToTensor(),
                                    #transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                    ]))

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.test_batch_size
                                             )

        
    images,seg = next(iter(train_dataloader))
    print("Images shape:",images.shape)
    print("Segmentation shape:",seg.shape)
    images_test = next(iter(test_dataloader))
    print("Images shape (test):",images_test.shape)

    # Visualize a few images
    if args.plot_samples and args.batch_size < 10:
        out = torchvision.utils.make_grid(seg)
        imshow(out,title='Algumas imagens')
        out2 = torchvision.utils.make_grid(images_test)
        imshow(out2,title='Algumas imagens test')


                
    # Semantic Segmentation Fully Convolutional Network offered by Pytorch
    # ConvNet as fixed feature extractor
    print("Loading CNN ResNet 101 ...")
    model_fcn = models.segmentation.fcn_resnet101(pretrained=True,
                                              progress=True,
                                              num_classes=21,
                                              aux_loss=None)

        
    d = models.vgg1

    tic = time.time()
    prediction = model_fcn(images)        # (batch size) x (num_classes) x (pixels_y) x (pixels_x)
    elapsed_time = time.time() - tic
    print("Elapsed time forward pass ResNet101:",elapsed_time," seconds")

    # Only the parameters of final layer are being optimized 
    optimizer = optim.SGD(model_conv.fc.parameters(),lr=args.lr,momentum=args.momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)

    # Training the model
    """
    model_conv = train_model(model_conv,
                             criterion,
                             optimizer,
                             exp_lr_scheduler,
                             args.num_epochs,
                             train_dataloader,
                             test_dataloader,
                             device)
    """
    

    if (args.save_model):
        torch.save(model.state_dict(),"fasion_mnist_cnn.pt")
        
 

## Run main
if __name__ == '__main__':
    main()
