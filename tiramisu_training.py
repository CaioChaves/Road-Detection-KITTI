from models import tiramisu
import torch
import argparse
import time
import os
from kitti_semantics_dataloader import KittiDataset
import torchvision
import torchvision.transforms as transforms
from datasets import camvid
from datasets import joint_transforms
import utils.imgs
import utils.training as train_utils
import torch.nn as nn
from patchCNN_tools import SecondsToText, PlotAccuracyEvolution, PlotLossEvolution, WriteInfoFile, AccuracyTest, WriteCSVFile
from progressbar import ProgressBar, Percentage, Bar, ETA, AdaptiveETA
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Tiramisu Semantic Segmentation CNN for KittiDataset')
    parser.add_argument('--out-layer', type=str, default='softmax',
                        help='Last layer on the model architecture (options: softmax (default) or sigmoid')
    parser.add_argument('--criterion', type=str, default='nll',
                        help='Loss function to be used (options: nll or bce; defatul: nll)')
    parser.add_argument('--threshold', type=float, default=0.5, metavar='N',
                        help='threshold value for class separation between 0.0 and 1.0 (default: 0.5)')
    parser.add_argument('--num-classes', type=int, default=1, 
                        help='number of classes in the output')
    parser.add_argument('--data-split', type=float, default=0.8, 
                        help='fraction of data for train dataset (default: 0.8). (1-data_split) is the fraction of validation dataset')                    
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4} if use_cuda else {}
    currDir = os.getcwd()

    # Dataset
    dataset = KittiDataset(currDir,
                        split = 'train',
                        target_type = 'semantic_binary_ground',
                        balanced=False,
                        colormodel = 'RGB', 
                        transform = transforms.Compose([
                                     transforms.ToTensor(),
                                     #transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
                                     ]))

    #train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset,[int(len(train_dataset)*args.data_split),int(len(train_dataset)*round(1.0-args.data_split,2))])
    training_indices = [int(line.strip()) for line in open('training_indices.txt')]
    validation_indices = [int(line.strip()) for line in open('validation_indices.txt')]
   
    train_dataset = torch.utils.data.Subset(dataset,training_indices)
    validation_dataset = torch.utils.data.Subset(dataset,validation_indices)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=args.batch_size
                                                 ,**kwargs
                                                 )

    validation_dataloader = torch.utils.data.DataLoader(validation_dataset,
                                                 batch_size=args.batch_size
                                                 ,**kwargs
                                                 )

    images,seg = next(iter(train_dataloader))
    print("Training Images Batch shape:",images.shape)
    print("Training Segmentation Batch shape:",seg.shape)
    
    images_val,seg_val = next(iter(validation_dataloader))
    print("Validation Images Batch shape:",images_val.shape)
    print("Validation Segmentation Batch shape:",seg_val.shape)
    
    LR = 0.0001
    LR_DECAY = 0.995
    DECAY_EVERY_N_EPOCHS = 1
    N_EPOCHS = args.epochs
    torch.cuda.manual_seed(0)
    model = tiramisu.FCDenseNet57(n_classes=args.num_classes,outfunc=args.out_layer).to(device)
    model.apply(train_utils.weights_init)
    model_learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of model learnable parameters: ',model_learnable_parameters)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
    
    #criterion = nn.NLLLoss2d(weight=camvid.class_weight.to(device)).to(device)
    if args.criterion == 'nll':
        criterion = nn.NLLLoss2d().to(device)
    if args.criterion == 'bce':
        criterion = nn.BCELoss().to(device)

    average_tr_loss_pile = []
    ACC_tr_loss_pile = []
    average_val_loss_pile = []
    ACC_val_loss_pile = []

    # Create directory to save results
    if os.path.isdir(currDir+'/resultsTiramisu'):
        os.chdir(currDir+'/resultsTiramisu')
    else: 
        os.mkdir(currDir+'/resultsTiramisu')
        os.chdir(currDir+'/resultsTiramisu')
    name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    os.mkdir(name)
    os.chdir(os.getcwd()+'/'+name)
    os.mkdir('patch_cnn_models')
    os.chdir(os.getcwd()+'/patch_cnn_models')

    # Progress Bar
    widgets = [Percentage(),' ',Bar(),' ',AdaptiveETA()]
    pbar = ProgressBar(widgets=widgets,maxval=200/args.batch_size)
    pbar.start()

    for epoch in pbar(range(1, args.epochs + 1)):
        since = time.time()
        trn_loss, trn_err = train_utils.train(args, model, train_dataloader, optimizer, criterion, device, epoch) 
        print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f} %'.format(epoch, trn_loss, 100*trn_err))
        time_elapsed = time.time() - since
        print('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        val_loss, val_err = train_utils.test(args, model, validation_dataloader, criterion, device, epoch)   #### MUDAR para validation data loader
        print('Val - Loss: {:.4f} | Acc: {:.4f} %'.format(val_loss, 100*val_err))
        time_elapsed = time.time() - since
        print('Total Time {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
        #train_utils.save_weights(model, epoch, val_loss, val_err)
        train_utils.adjust_learning_rate(LR, LR_DECAY, optimizer, epoch, DECAY_EVERY_N_EPOCHS)
        average_tr_loss_pile.append(trn_loss)
        ACC_tr_loss_pile.append(round(100*trn_err,2))
        average_val_loss_pile.append(val_loss)
        ACC_val_loss_pile.append(round(100*val_err,2))

        if (args.save_model):
            torch.save(model.state_dict(),'epoch_{}_patch_cnn_model.pth'.format(epoch))

    # Save model, plot results and write log file
    os.chdir(currDir+'/resultsTiramisu'+'/'+name)
    elapsed_time = time.time() - since
    WriteInfoFile(name,0,0,0,args.data_split,args.batch_size,args.epochs,optimizer,SecondsToText(elapsed_time),criterion,model,model_param=model_learnable_parameters)
    WriteCSVFile(args.epochs,average_tr_loss_pile,average_val_loss_pile,ACC_tr_loss_pile,ACC_val_loss_pile)
    PlotLossEvolution(average_tr_loss_pile,average_val_loss_pile)
    PlotAccuracyEvolution(ACC_tr_loss_pile,ACC_val_loss_pile)  
        
    os.chdir(currDir)

if __name__ == '__main__':
    main()