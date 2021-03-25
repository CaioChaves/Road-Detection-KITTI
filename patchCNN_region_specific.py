import argparse
import torch
from kitti_semantics_dataloader import KittiDataset
import torchvision
import torchvision.transforms as transforms
import os
from model_archicteture import MyNetwork
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import time
import torch.utils.tensorboard as TensorBoard    
from patchCNN_tools import SecondsToText, PlotAccuracyEvolution, PlotLossEvolution, WriteInfoFile, AccuracyTest, PlotConfusionMatrix, WriteCSVFile, WriteCSVFile2, GetCropLimits
from patchCNN_tools import PrePlotEvolution, Convert2CenteredCoordinates
                                                                                                                                               
def train(args, model, device, train_dataloader, optimizer, patch_size, ppi, loss_func, Rlim, Clim):
    model.train()
    LossPile = []
    confusion_matrix = np.zeros([2,2])
    correct, total = 0,0
    for _, (original, segmentation) in enumerate(train_dataloader):
        for _ in range(ppi):
            patch_x_pos = random.randint(Clim[0],Clim[1]-patch_size)
            patch_y_pos = random.randint(Rlim[0],Rlim[1]-patch_size)
            original_patch = original[:,:,patch_y_pos:patch_y_pos+patch_size,patch_x_pos:patch_x_pos+patch_size]
            segmentation_patch = segmentation[:,:,patch_y_pos:patch_y_pos+patch_size,patch_x_pos:patch_x_pos+patch_size]
            original_patch, segmentation_patch = original_patch.to(device), segmentation_patch.to(device)
            optimizer.zero_grad()
           
            pho, theta = Convert2CenteredCoordinates(patch_y_pos,patch_x_pos,heigth=original.shape[2],width=original.shape[3])
            if args.model_arch == 'base':
                output = model(original_patch)                
            elif 'mini':
                output = model(original_patch,device=device,pho=pho,theta=theta)

            target = segmentation_patch.contiguous().view(len(original),patch_size*patch_size*1).median(dim=1).values.view_as(output).to(device)
            loss = loss_func(output,target)
            loss.backward()
            optimizer.step()
            pos_cm, increment = AccuracyTest(output,target)
            for item in pos_cm:
                confusion_matrix[item[0]][item[1]] += 1            
            correct += increment
            total += original.shape[0]      # batch size
            LossPile.append(loss.item())
    return np.mean(LossPile), correct, total, confusion_matrix


def validate(args, model, device, validation_dataloader, patch_size, ppi, loss_func, Rlim, Clim):
    model.eval()
    with torch.no_grad():
        LossPile = []
        confusion_matrix = np.zeros([2,2])
        correct, total = 0,0
        for _, (original, segmentation) in enumerate(validation_dataloader):
            for _ in range(ppi):
                patch_x_pos = random.randint(Clim[0],Clim[1]-patch_size)
                patch_y_pos = random.randint(Rlim[0],Rlim[1]-patch_size)
                original_patch = original[:,:,patch_y_pos:patch_y_pos+patch_size,patch_x_pos:patch_x_pos+patch_size]
                segmentation_patch = segmentation[:,:,patch_y_pos:patch_y_pos+patch_size,patch_x_pos:patch_x_pos+patch_size]
                original_patch, segmentation_patch = original_patch.to(device), segmentation_patch.to(device)
                
                pho, theta = Convert2CenteredCoordinates(patch_y_pos,patch_x_pos,heigth=original.shape[2],width=original.shape[3])
                if args.model_arch == 'base':
                    output = model(original_patch)                
                elif 'mini':
                    output = model(original_patch,device=device,pho=pho,theta=theta)              
                target = segmentation_patch.contiguous().view(len(original),patch_size*patch_size*1).median(dim=1).values.view_as(output).to(device)
                
                loss = loss_func(output,target)
                pos_cm, increment = AccuracyTest(output,target)
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
    parser.add_argument('--ppi', type=int, default=1000, metavar='N',
                        help='patchs per image (default: 1000)')
    parser.add_argument('--data-split', type=float, default=0.8, 
                        help='fraction of data for train dataset (default: 0.8). (1-data_split) is the fraction of validation dataset')                    
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--model-folder',type=str,dest='model_folder',
                        help='Folder name containing model (.pt)') 
    parser.add_argument('--epoch-number', type=int, 
                        help='input epoch number (default=none)')
    parser.add_argument('--split-in', type=list, default = [2,3],
                        help='list of regions to slit the images (default=23 (2 rows 2 columns  )))')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--arch', type=str,dest='model_arch',default='base',
                        help='string to identify which CNN architecture to use (options: base (default) or mini)')
    parser.add_argument('--spatial-feats', action='store_true', default=False,
                        help='Include position features to CNN (def. False)')
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
        

    ModelList = []
    OptimizerList = []
    L = []
    O = []
    modelDir = str(os.getcwd()+'/results/'+args.model_folder+'/patch_cnn_models/epoch_{}_patch_cnn_model.pth'.format(args.epoch_number))

    total_regions_row, total_regions_column = [int(args.split_in[0]),int(args.split_in[1])]

    for _ in range(0,total_regions_row):
        for _ in range(0,total_regions_column):
            model = MyNetwork(num_classes_out=1)
            model.load_state_dict(torch.load(modelDir))
            model.to(device)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
            L.append(model)
            O.append(optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum))
        ModelList.append(L)
        OptimizerList.append(O)
    
    loss_func = torch.nn.BCELoss()
    ppi = args.ppi
    average_tr_loss_pile = []
    ACC_tr_loss_pile = []
    average_val_loss_pile = []
    ACC_val_loss_pile = []
    CM_train_pile = []
    CM_val_pile = []
    auxLoss1, auxLoss2, auxAcc1, auxAcc2, auxCM1, auxCM2 = [],[],[],[],[],[]  # contains information for all epochs of a region [x,y]
    auxLoss3, auxLoss4, auxAcc3, auxAcc4, auxCM3, auxCM4 = [],[],[],[],[],[]
 
    # Create directory to save results
    if args.save_model:
        os.chdir(currDir+'/results')
        name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        os.mkdir(name)
        os.chdir(os.getcwd()+'/'+name)
        os.mkdir('patch_cnn_models')
        os.chdir(os.getcwd()+'/patch_cnn_models')

    for region_row in range(0,total_regions_row):
        for region_column in range(0,total_regions_column):
            Rlim, Clim = GetCropLimits([region_row,region_column],division=[total_regions_row,total_regions_column],img_size=[370,1224])
            for epoch in range(1, args.epochs + 1):
                average_tr_loss, ACC_correct_train, ACC_total_train, confusion_matrix = train(args, ModelList[region_row][region_column], device, train_dataloader, 
                                                                                        OptimizerList[region_row][region_column], args.patch_size, ppi, loss_func, 
                                                                                        Rlim,Clim)
                auxLoss1.append(average_tr_loss)
                auxAcc1.append(round(100*ACC_correct_train/ACC_total_train,2))
                auxCM1.append(confusion_matrix)
                print('TRAINING: \t Epoch: {} \t Region: {},{} \t Average Loss: {:.4f} \t Accuracy: {}/{} ({}%)'.
                            format(epoch, region_row, region_column, average_tr_loss, ACC_correct_train, ACC_total_train, round(100*ACC_correct_train/ACC_total_train,2)))
                average_val_loss, ACC_correct_val, ACC_total_val, confusion_matrix = validate(args, ModelList[region_row][region_column], device, validation_dataloader,
                                                                                        args.patch_size, ppi, loss_func, Rlim, Clim)
                auxLoss2.append(average_val_loss)
                auxAcc2.append(round(100*ACC_correct_val/ACC_total_val,2))
                print('VALIDATION: \t Epoch: {} \t Region: {},{} \t Average Loss: {:.4f} \t Accuracy: {}/{} ({}%)'.
                            format(epoch, region_row, region_column, average_val_loss, ACC_correct_val,ACC_total_val,round(100*ACC_correct_val/ACC_total_val,2)))
                auxCM2.append(confusion_matrix)
                if (args.save_model):
                    torch.save(model.state_dict(),'epoch_{}_region_{}_{}_patch_cnn_model.pth'.format(epoch,region_row,region_column))
            auxLoss3.append(auxLoss1)
            auxAcc3.append(auxAcc1)            
            auxLoss4.append(auxLoss2)
            auxAcc4.append(auxAcc2)
            auxCM3.append(auxCM1)
            auxCM4.append(auxCM2)
            auxLoss1, auxLoss2, auxAcc1, auxAcc2, auxCM1, auxCM2 = [],[],[],[],[],[]
        average_tr_loss_pile.append(auxLoss3)
        ACC_tr_loss_pile.append(auxAcc3)            
        average_val_loss_pile.append(auxLoss4)
        ACC_val_loss_pile.append(auxAcc4)
        CM_train_pile.append(auxCM3)
        CM_val_pile.append(auxCM4)
        auxLoss3, auxLoss4, auxAcc3, auxAcc4, auxCM3, auxCM4 = [],[],[],[],[],[]

    # Save model, plot results and write log file
    if not (args.save_model):
        os.chdir(currDir+'/results')
        name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        os.mkdir(name)

    os.chdir(currDir+'/results'+'/'+name)
    elapsed_time = time.time() - start_time
    WriteInfoFile(name,args.patch_size,ppi,0,args.data_split,args.batch_size,args.epochs,optimizer,SecondsToText(elapsed_time),loss_func,model,args.split_in)
    WriteCSVFile2(args.epochs,average_tr_loss_pile,average_val_loss_pile,ACC_tr_loss_pile,ACC_val_loss_pile,total_regions_row,total_regions_column)
    # Preplotevolution calls both the loss function plot and the accuracy plot
    PrePlotEvolution(average_tr_loss_pile,average_val_loss_pile,ACC_tr_loss_pile,ACC_val_loss_pile,total_regions_row,total_regions_column)
    
    """# Confusion matrix
    dir_name_cm_training = 'confusion_matrix_training'
    if os.path.isdir(dir_name_cm_training):
        os.chdir(dir_name_cm_training)
    else: 
        os.mkdir(dir_name_cm_training)
        os.chdir(dir_name_cm_training)
    for j in range(total_regions_row):
        for k in range(total_regions_column):
            region_name = 'region_{}_{}'.format(j,k)
            os.mkdir(region_name)
            os.chdir(region_name)
            #PrePlotConfusionMatrix(CM_train_pile,CM_val_pile,total_regions_row,total_regions_column)
            for idx, elem in enumerate(CM_train_pile[:][j][k]):
                PlotConfusionMatrix(idx,elem,normalize=True,cmap=plt.cm.Greys,phase='Training')
            os.chdir('../')
    os.chdir(currDir+'/results'+'/'+name)

    dir_name_cm_val = 'confusion_matrix_validation'
    if os.path.isdir(dir_name_cm_val):
        os.chdir(dir_name_cm_val)
    else: 
        os.mkdir(dir_name_cm_val)
        os.chdir(dir_name_cm_val)
    for j in range(total_regions_row):
        for k in range(total_regions_column):
            region_name = 'region_{}_{}'.format(j,k)
            os.mkdir(region_name)
            os.chdir(region_name)   
            for idx, elem in enumerate(CM_val_pile[:][j][k]):
                PlotConfusionMatrix(idx,elem,normalize=True,cmap=plt.cm.Greys,phase='Validation')
            os.chdir('../')
    os.chdir(currDir+'/results'+'/'+name)"""
    
    os.chdir(currDir)

    

if __name__ == '__main__':
    main()