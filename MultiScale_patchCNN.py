import argparse
import torch
from kitti_semantics_dataloader import KittiDataset
import torchvision
import torchvision.transforms as transforms
import os
from model_archicteture import MyNetwork, MultiScalePatchNet
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import time
from patchCNN_tools import SecondsToText, PlotAccuracyEvolution, PlotLossEvolution, WriteInfoFile, AccuracyTest, PlotConfusionMatrix, WriteCSVFile, Convert2PolarCoordinates
from patchCNN_tools import Convert2CenteredCoordinates
from progressbar import ProgressBar, Percentage, Bar, ETA, AdaptiveETA

def train(args, model, device, train_dataloader, optimizer, epoch, patch_size, ppi, loss_func):
    model.train()
    LossPile = []
    confusion_matrix = np.zeros([2,2])
    correct, total = 0,0
    for _, (original, segmentation) in enumerate(train_dataloader):
        for _ in range(ppi):
            patch_x_pos = random.randint(patch_size,original.shape[3]-patch_size)
            patch_y_pos = random.randint(patch_size,original.shape[2]-patch_size)
        
            # Multi-scale patches (14x14 28x28 56x56)
            small_patch = original[:,:,patch_y_pos-int(patch_size/4):patch_y_pos+int(patch_size/4),
                                   patch_x_pos-int(patch_size/4):patch_x_pos+int(patch_size/4)].to(device)
            medium_patch = original[:,:,patch_y_pos-int(patch_size/2):patch_y_pos+int(patch_size/2),
                                   patch_x_pos-int(patch_size/2):patch_x_pos+int(patch_size/2)].to(device)
            large_patch = original[:,:,patch_y_pos-patch_size:patch_y_pos+patch_size,
                                    patch_x_pos-patch_size:patch_x_pos+patch_size].to(device)
            
            optimizer.zero_grad()

            if args.spatial_feats:
                pho, theta = Convert2CenteredCoordinates(patch_y_pos,patch_x_pos,heigth=original.shape[2],width=original.shape[3])
                prediction = model(small_patch,medium_patch,large_patch,device,pho,theta)
            else: 
                prediction = model(small_patch,medium_patch,large_patch,device)

            # Segmentatoin
            segmentation_small_patch = segmentation[:,:,patch_y_pos-int(patch_size/4):patch_y_pos+int(patch_size/4),
                                                    patch_x_pos-int(patch_size/4):patch_x_pos+int(patch_size/4)].to(device)
            segmentation_medium_patch = segmentation[:,:,patch_y_pos-int(patch_size/2):patch_y_pos+int(patch_size/2),
                                                    patch_x_pos-int(patch_size/2):patch_x_pos+int(patch_size/2)].to(device)
            segmentation_large_patch = segmentation[:,:,patch_y_pos-patch_size:patch_y_pos+patch_size,
                                                    patch_x_pos-patch_size:patch_x_pos+patch_size].to(device)

            # Target
            s = segmentation_small_patch.contiguous().view(len(original),int(patch_size*patch_size/4)).median(dim=1).values.reshape(1,len(original))
            m = segmentation_medium_patch.contiguous().view(len(original),int(patch_size*patch_size*1)).median(dim=1).values.reshape(1,len(original))
            l = segmentation_large_patch.contiguous().view(len(original),int(patch_size*patch_size*4)).median(dim=1).values.reshape(1,len(original))   
            
            target = torch.cat([s,m,l],dim=0).median(dim=0).values.view_as(prediction)
            loss = loss_func(prediction,target)
            loss.backward()
            optimizer.step()
            pos_cm, increment = AccuracyTest(prediction,target)
            for item in pos_cm:
                confusion_matrix[item[0]][item[1]] += 1            
            correct += increment
            total += original.shape[0]      # batch size
            LossPile.append(loss.item())
    return np.mean(LossPile), correct, total, confusion_matrix


def validate(args, model, device, validation_dataloader, patch_size, ppi, loss_func):
    model.eval()
    with torch.no_grad():
        LossPile = []
        confusion_matrix = np.zeros([2,2])
        correct, total = 0,0
        for _, (original, segmentation) in enumerate(validation_dataloader):
            for _ in range(ppi):
                patch_x_pos = random.randint(patch_size,original.shape[3]-patch_size)
                patch_y_pos = random.randint(patch_size,original.shape[2]-patch_size)

                # Multi-scale patches (14x14 28x28 56x56)
                small_patch = original[:,:,patch_y_pos-int(patch_size/4):patch_y_pos+int(patch_size/4),
                                    patch_x_pos-int(patch_size/4):patch_x_pos+int(patch_size/4)].to(device)
                medium_patch = original[:,:,patch_y_pos-int(patch_size/2):patch_y_pos+int(patch_size/2),
                                    patch_x_pos-int(patch_size/2):patch_x_pos+int(patch_size/2)].to(device)
                large_patch = original[:,:,patch_y_pos-patch_size:patch_y_pos+patch_size,
                                        patch_x_pos-patch_size:patch_x_pos+patch_size].to(device)
                    
                segmentation_small_patch = segmentation[:,:,patch_y_pos-int(patch_size/4):patch_y_pos+int(patch_size/4),
                                        patch_x_pos-int(patch_size/4):patch_x_pos+int(patch_size/4)].to(device)
                segmentation_medium_patch = segmentation[:,:,patch_y_pos-int(patch_size/2):patch_y_pos+int(patch_size/2),
                                        patch_x_pos-int(patch_size/2):patch_x_pos+int(patch_size/2)].to(device)
                segmentation_large_patch = segmentation[:,:,patch_y_pos-patch_size:patch_y_pos+patch_size,
                                        patch_x_pos-patch_size:patch_x_pos+patch_size].to(device)

                if args.spatial_feats:
                    pho, theta = Convert2CenteredCoordinates(patch_y_pos,patch_x_pos,heigth=original.shape[2],width=original.shape[3])
                    prediction = model(small_patch,medium_patch,large_patch,device,pho,theta)
                else: 
                    prediction = model(small_patch,medium_patch,large_patch,device)

                # Target
                s = segmentation_small_patch.contiguous().view(len(original),int(patch_size*patch_size/4)).median(dim=1).values.reshape(1,len(original))
                m = segmentation_medium_patch.contiguous().view(len(original),int(patch_size*patch_size*1)).median(dim=1).values.reshape(1,len(original))
                l = segmentation_large_patch.contiguous().view(len(original),int(patch_size*patch_size*4)).median(dim=1).values.reshape(1,len(original))      
                target = torch.cat([s,m,l],dim=0).median(dim=0).values.view_as(prediction)
                loss = loss_func(prediction,target)
                pos_cm, increment = AccuracyTest(prediction,target)
                for item in pos_cm:
                    confusion_matrix[item[0]][item[1]] += 1
                correct += increment
                total += original.shape[0]      # batch size
                LossPile.append(loss.item())
    return np.mean(LossPile), correct, total, confusion_matrix

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Patch-based CNN')
    parser.add_argument('--patch-size', type=int, default=28, metavar='N',
                        help='input patch size for crooping image (default: 28)')
    parser.add_argument('--ppi-coeff', type=float, default=0.01, metavar='PPI',
                        help='coefficient related to the number of random selected patches per image (default: 0.01)')
    parser.add_argument('--ppi', type=int, default=1000, metavar='N',
                        help='patchs per image (default: 1000)')
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
    parser.add_argument('--balanced', action='store_true', default=False,
                        help='For using a dataset in which the amount of pixels tagged ground is equal to the ones tagged non-ground. (def. False)')
    parser.add_argument('--spatial-feats', action='store_true', default=False,
                        help='Include position features to CNN (def. False)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4} if use_cuda else {}
    start_time = time.time()
    currDir = os.getcwd()

    # Dataset
    dataset = KittiDataset(currDir,
                        split = 'train',
                        target_type = 'semantic_binary_ground',
                        balanced=args.balanced,
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
    
    print('train_dataset length:',len(train_dataset))
    print('validation_dataset length:',len(validation_dataset))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)

    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size ,**kwargs)

    images,seg = next(iter(train_dataloader))
    print("Training Images Batch shape:",images.shape)
    print("Training Segmentation Batch shape:",seg.shape)
    
    images_val,seg_val = next(iter(validation_dataloader))
    print("Validation Images Batch shape:",images_val.shape)
    print("Validation Segmentation Batch shape:",seg_val.shape)

    model = MultiScalePatchNet(in_channels=3,num_classes_out=1,spatial_features=args.spatial_feats).to(device)
    #model = MyNetwork(in_channels=3,num_classes_out=1).to(device)
    model_learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of model learnable parameters: ',model_learnable_parameters)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    loss_func = torch.nn.BCELoss()
    ppi = args.ppi
    average_tr_loss_pile = []
    ACC_tr_loss_pile = []
    average_val_loss_pile = []
    ACC_val_loss_pile = []
    CM_train_pile = []
    CM_val_pile = []
    
    # Create directory to save results
    os.chdir(currDir+'/results')
    name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    os.mkdir(name)
    os.chdir(os.getcwd()+'/'+name)
    os.mkdir('patch_cnn_models')
    os.chdir(os.getcwd()+'/patch_cnn_models')

    # Progress Bar
    widgets = [Percentage(),' ',Bar(),' ',AdaptiveETA()]
    pbar = ProgressBar(widgets=widgets,maxval=ppi*200/args.batch_size)
    pbar.start()

    for epoch in pbar(range(1, args.epochs + 1)):
        average_tr_loss, ACC_correct_train, ACC_total_train, confusion_matrix = train(args, model, device, train_dataloader, optimizer, epoch, args.patch_size, ppi, loss_func)
        average_tr_loss_pile.append(average_tr_loss)
        ACC_tr_loss_pile.append(round(100*ACC_correct_train/ACC_total_train,2))
        CM_train_pile.append(confusion_matrix)
        print('TRAINING: Epoch: {} \t Average Loss: {:.4f} \t Accuracy: {}/{} ({}%)'.format(epoch, average_tr_loss, ACC_correct_train, ACC_total_train, round(100*ACC_correct_train/ACC_total_train,2)))
        average_val_loss, ACC_correct_val, ACC_total_val, confusion_matrix = validate(args, model, device, validation_dataloader, args.patch_size, ppi, loss_func) 
        average_val_loss_pile.append(average_val_loss)
        ACC_val_loss_pile.append(round(100*ACC_correct_val/ACC_total_val,2))
        print('VALIDATION: Epoch: {} \t Average Loss: {:.4f} \t Accuracy: {}/{} ({}%)'.format(epoch, average_val_loss, ACC_correct_val,ACC_total_val,round(100*ACC_correct_val/ACC_total_val,2)))
        CM_val_pile.append(confusion_matrix)
        if (args.save_model):
            torch.save(model.state_dict(),'epoch_{}_patch_cnn_model.pth'.format(epoch))

    # Save model, plot results and write log file
    os.chdir(currDir+'/results'+'/'+name)
    elapsed_time = time.time() - start_time
    WriteInfoFile(name,args.patch_size,ppi,args.ppi_coeff,args.data_split,args.batch_size,args.epochs,optimizer,SecondsToText(elapsed_time),loss_func,model,model_param=model_learnable_parameters)
    WriteCSVFile(args.epochs,average_tr_loss_pile,average_val_loss_pile,ACC_tr_loss_pile,ACC_val_loss_pile)
    PlotLossEvolution(average_tr_loss_pile,average_val_loss_pile)
    PlotAccuracyEvolution(ACC_tr_loss_pile,ACC_val_loss_pile)  
    
    # Confusion matrix
    dir_name_cm_training = 'confusion_matrix_training'
    if os.path.isdir(dir_name_cm_training):
        os.chdir(dir_name_cm_training)
    else: 
        os.mkdir(dir_name_cm_training)
        os.chdir(dir_name_cm_training)
    for idx, elem in enumerate(CM_train_pile):
        PlotConfusionMatrix(idx,elem,normalize=True,cmap=plt.cm.Greys,phase='Training')
    os.chdir(currDir+'/results'+'/'+name)

    dir_name_cm_val = 'confusion_matrix_validation'
    if os.path.isdir(dir_name_cm_val):
        os.chdir(dir_name_cm_val)
    else: 
        os.mkdir(dir_name_cm_val)
        os.chdir(dir_name_cm_val)
    for idx, elem in enumerate(CM_val_pile):
        PlotConfusionMatrix(idx,elem,normalize=True,cmap=plt.cm.Greys,phase='Validation')

    os.chdir(currDir)


if __name__ == '__main__':
    main()