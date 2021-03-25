import argparse
import torch
import time
from kitti_semantics_dataloader import KittiDataset
import torchvision
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from datetime import datetime
import time
from patchCNN_tools import WriteInfoFile, WriteCSVFile, SecondsToText, PlotLossEvolution, PlotAccuracyEvolution, PlotConfusionMatrix
from progressbar import ProgressBar, Percentage, Bar, ETA, AdaptiveETA
import torch.optim as optim
from patchCNN_tools import CalculateIoU, WriteInfoFile3

def train(args, model, device, train_dataloader, optimizer, epoch, loss_func):
    model.train()
    LossPile = []
    IOUPile = []
    for _, (original, segmentation) in enumerate(train_dataloader):
        original, segmentation = original.to(device), segmentation.to(device)
        optimizer.zero_grad()
        prediction = model(original)
        prediction = prediction.get('out')
        loss = loss_func(prediction,segmentation)
        loss.backward()
        optimizer.step()
        LossPile.append(loss.item())
        iou_score = CalculateIoU(target=segmentation,prediction=prediction,threshold=args.threshold)
        IOUPile.append(iou_score)
    return np.mean(LossPile), np.mean(IOUPile)

def validate(args, model, device, validation_dataloader, loss_func):
    model.eval()
    LossPile = []
    IOUPile = []
    for _, (original, segmentation) in enumerate(validation_dataloader):
        original, segmentation = original.to(device), segmentation.to(device)
        prediction = model(original)
        prediction = prediction.get('out')
        loss = loss_func(prediction,segmentation)
        LossPile.append(loss.item())
        iou_score = CalculateIoU(target=segmentation,prediction=prediction,threshold=args.threshold)
        IOUPile.append(iou_score)
    return np.mean(LossPile), np.mean(IOUPile)
  
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='FCN Semantic Segmentation')
    parser.add_argument('--num-classes', type=int, default=1,
                        help='Output number of classes (default:1)')
    parser.add_argument('--threshold', type=float, default=0.5, metavar='N',
                        help='threshold value for class separation between 0.0 and 1.0 (default: 0.5)')
    parser.add_argument('--pre-trained', action='store_true', default=False,
                        help='load torchvision pre-trained model')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--data-split', type=float, default=0.8, 
                        help='fraction of data for train dataset (default: 0.8). (1-data_split) is the fraction of validation dataset')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    ## Import dataset and dataloader
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4} if use_cuda else {}
    start_time = time.time()
    currDir = os.getcwd()

    # Dataset
    train_dataset = KittiDataset(currDir,
                        split = 'train',
                        target_type = 'semantic_binary_ground',
                        transform = transforms.Compose([
                                     transforms.ToTensor(),
                                     #transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
                                     ]))

    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset,[int(len(train_dataset)*args.data_split),int(len(train_dataset)*round(1.0-args.data_split,2))])

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

    ## Chosse torchvision segmentation model and adapt it to our case
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=args.pre_trained,progress=True,num_classes=args.num_classes)
    new_classifier = nn.Sequential(OrderedDict([('conv1',nn.Conv2d(2048, 512, 3, padding=1, bias=False)),
                                            ('bn1',nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)),
                                            ('relu',nn.ReLU()),
                                            ('drop',nn.Dropout(0.1)),
                                            ('conv2',nn.Conv2d(512,args.num_classes,kernel_size=(1,1),stride=(1,1))),
                                            ('soft',nn.Softmax2d())
                                            ]))

    model.classifier = new_classifier
    model.to(device)
    model_learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of learnable parameters:'+str(model_learnable_parameters))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    loss_func = torch.nn.BCELoss()
    average_tr_loss_pile = []
    ACC_tr_loss_pile = []
    average_val_loss_pile = []
    ACC_val_loss_pile = []

    average_tr_iou_pile = []
    average_val_iou_pile = []
    
    # Create directory to save results
    os.chdir(currDir+'/resultsFCN')
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
        average_tr_loss, average_tr_iou = train(args, model, device, train_dataloader, optimizer, epoch, loss_func)
        average_tr_loss_pile.append(average_tr_loss)
        average_tr_iou_pile.append(average_tr_iou)
        #ACC_tr_loss_pile.append(round(100*ACC_correct_train/ACC_total_train,2))
        print('TRAINING: Epoch: {} \t Average Loss: {:.4f} \t Accuracy: {}%'.format(epoch, average_tr_loss, round(100*average_tr_iou,2)))
        #print('TRAINING: Epoch: {} \t Average Loss: {:.4f} \t Accuracy: {}/{} ({}%)'.format(epoch, average_tr_loss, ACC_correct_train, ACC_total_train, round(100*ACC_correct_train/ACC_total_train,2)))
        average_val_loss, average_val_iou = validate(args, model, device, validation_dataloader, loss_func) 
        average_val_loss_pile.append(average_val_loss)
        average_val_iou_pile.append(average_val_iou)
        #ACC_val_loss_pile.append(round(100*ACC_correct_val/ACC_total_val,2))
        print('VALIDATION: Epoch: {} \t Average Loss: {:.4f} \t Accuracy: {}%'.format(epoch, average_val_loss, round(100*average_val_iou,2)))
        #print('VALIDATION: Epoch: {} \t Average Loss: {:.4f} \t Accuracy: {}/{} ({}%)'.format(epoch, average_val_loss, ACC_correct_val,ACC_total_val,round(100*ACC_correct_val/ACC_total_val,2)))
        
    if (args.save_model):
        torch.save(model.state_dict(),'epoch_{}_fcn_model.pth'.format(epoch))

    # Save model, plot results and write log file
    os.chdir(currDir+'/resultsFCN'+'/'+name)
    elapsed_time = time.time() - start_time
    WriteInfoFile3(name,elapsed_time,loss_func, optimizer, model, args.pre_trained, model_learnable_parameters)
    WriteCSVFile(args.epochs,average_tr_loss_pile, average_tr_iou_pile,average_val_loss_pile,average_val_iou_pile)
    PlotLossEvolution(average_tr_loss_pile,average_val_loss_pile)
    PlotAccuracyEvolution(average_tr_iou_pile,average_val_iou_pile)
   
    os.chdir(currDir)


    

if __name__ == '__main__':
    main()