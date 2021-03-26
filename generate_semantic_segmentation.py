import argparse
import torch
import torchvision
from model_archicteture import MyNetwork, MultiScalePatchNet, MinimalisticCNN
import os
from kitti_semantics_dataloader import KittiDataset
import torchvision.transforms as transforms
import random
import time
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from patchCNN_tools import WriteInfoFile2, SecondsToText, CalculateIoU, Convert2CenteredCoordinates
from PIL import Image
from progressbar import ProgressBar, Percentage, Bar, ETA, AdaptiveETA
from sklearn.metrics import precision_recall_curve, average_precision_score
from inspect import signature
from models import tiramisu


"""
Generate semantic segmentation:
This code uses the patch-based CNN trained over the KittiDataset
to assign a class-label for each pixel.
Classes available are: Navigable ground and Non-navigable ground
INPUT: Pre-trained model folder
       Name of the image file to semantic segmentation (3x370x1220)
OUTPUT: Segmentation map: (1x370x1220)
"""

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Generate semantic segmentation')
    parser.add_argument('--arch', type=str,dest='model_arch',default='base',
                        help='string to identify which CNN architecture to use (options: base (default), mini, MS, tiramisu)')
    parser.add_argument('--patch-size', type=int, default=28, metavar='N',
                        help='input patch size for crooping image (default: 28)')
    parser.add_argument('--img-number', type=int, default=1000,
                        help='input image number from 0 to 199 (default: random)')
    parser.add_argument('--stride', type=int, default=1,
                        help='step value between sucessive patchs (default: 1)')
    parser.add_argument('--epoch-number', type=int, 
                        help='input epoch number (default=none)')
    parser.add_argument('--model-folder',type=str,dest='model_folder',
                        help='Folder name containing model (.pt)')
    parser.add_argument('--threshold', type=float, default=0.5, metavar='N',
                        help='threshold value for class separation between 0.0 and 1.0 (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dataset', type=str, default='training',
                        help='String indicating which folder to pick image from (options: testing, training / default: training)')
    parser.add_argument('--spatial-feats', action='store_true', default=False,
                        help='Include position features to CNN (def. False)')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4} if use_cuda else {}
    currDir = os.getcwd()
  
    # Load model
    if args.model_arch == 'base':
        model = MyNetwork()
        model.load_state_dict(torch.load(os.getcwd()+'/results/'+args.model_folder+'/patch_cnn_models/epoch_{}_patch_cnn_model.pth'.format(args.epoch_number)))
    if args.model_arch == 'mini':
        model = MinimalisticCNN(spatial_features=args.spatial_feats)
        model.load_state_dict(torch.load(os.getcwd()+'/results/'+args.model_folder+'/patch_cnn_models/epoch_{}_patch_cnn_model.pth'.format(args.epoch_number)))
    if args.model_arch == 'MS':
        model = MultiScalePatchNet(spatial_features=args.spatial_feats)
        model.load_state_dict(torch.load(os.getcwd()+'/results/'+args.model_folder+'/patch_cnn_models/epoch_{}_patch_cnn_model.pth'.format(args.epoch_number)))
    if args.model_arch == 'tiramisu':
        model = tiramisu.FCDenseNet57(n_classes=1,outfunc='sigmoid')
        model.load_state_dict(torch.load(os.getcwd()+'/resultsTiramisu/'+args.model_folder+'/patch_cnn_models/epoch_{}_patch_cnn_model.pth'.format(args.epoch_number)))

    model.to(device)
    model.eval()
    with torch.no_grad():
        # Dataset
        if 0 <= args.img_number <= 999:
            img_number = args.img_number 
            print('Chosen image number:',img_number)
        else:
            img_number = random.randint(0,199)
            print('Randomly picked image number:',img_number)
        if 0 <= args.img_number <= 9:
                img_name = '00000'+str(img_number)+'_10.png'
        elif 10 <= args.img_number <= 99:
                img_name = '0000'+str(img_number)+'_10.png'
        elif 100 <= args.img_number <=999:
                img_name = '000'+str(img_number)+'_10.png'
        
        ## Loading image
        rootDir = os.getcwd()
        os.chdir(rootDir+'/data_semantics/'+str(args.dataset)+'/image_2')
        transToTensor = torchvision.transforms.ToTensor()
        transToPILImage = torchvision.transforms.ToPILImage()
        image = transToTensor(Image.open(img_name,mode='r'))
        prediction = torch.zeros(image.shape[1],image.shape[2])
        # Loop Scanning
        widgets = [Percentage(),' ',Bar(),' ',AdaptiveETA()]
        pbar = ProgressBar(widgets=widgets)
        range_row = range(int(args.patch_size/1),image.shape[1]-int(args.patch_size/1),args.stride)
        range_column = range(int(args.patch_size/1),image.shape[2]-int(args.patch_size/1),args.stride)
        pbar.start()
        start_time = time.time()
        if args.model_arch != 'tiramisu':
            for row in pbar(range_row):
                for column in range_column:
                    if args.model_arch == 'base' or 'mini':
                        medium_patch = image[:,row-int(args.patch_size/2):row+int(args.patch_size/2),column-int(args.patch_size/2):column+int(args.patch_size/2)]
                        medium_patch = medium_patch.view(1,3,args.patch_size,args.patch_size)
                    if args.spatial_feats and args.model_arch == 'mini':
                        pho, theta = Convert2CenteredCoordinates(column,row,heigth=image.shape[1],width=image.shape[2]) ## Spatial - features
                        P = model(medium_patch.to(device),device ,pho,theta)
                    if not args.spatial_feats and args.model_arch == 'mini':
                        P = model(medium_patch.to(device),device)
                    if args.model_arch == 'base':                    
                        P = model(medium_patch.to(device))

                    if args.model_arch == 'MS':
                        small_patch = image[:,row-int(args.patch_size/4):row+int(args.patch_size/4),column-int(args.patch_size/4):column+int(args.patch_size/4)]
                        small_patch = small_patch.view(1,3,int(args.patch_size/2),int(args.patch_size/2))

                        medium_patch = image[:,row-int(args.patch_size/2):row+int(args.patch_size/2),column-int(args.patch_size/2):column+int(args.patch_size/2)]
                        medium_patch = medium_patch.view(1,3,args.patch_size,args.patch_size)

                        large_patch = image[:,row-int(args.patch_size/1):row+int(args.patch_size/1),column-int(args.patch_size/1):column+int(args.patch_size/1)]
                        large_patch = large_patch.view(1,3,int(args.patch_size*2),int(args.patch_size*2))

                        pho, theta = Convert2CenteredCoordinates(column,row,heigth=image.shape[1],width=image.shape[2]) ## Spatial - features
                        P = model(small_patch.to(device),medium_patch.to(device),large_patch.to(device),device,pho,theta)

                    row_sup = int(np.ceil(args.stride/2))
                    row_inf = int(np.floor(args.stride/2))
                    column_esq = int(np.floor(args.stride/2))
                    column_dir = int(np.ceil(args.stride/2))

                    if row == range_row[0]:                 # top edge
                        row_inf = int(np.floor(args.patch_size))
                        if column == range_column[0]:       #  top left corner
                            column_esq = int(np.floor(args.patch_size))
                        if column == range_column[-1]:      #  top right corner
                            column_dir = int(np.floor(args.patch_size))               # Until the end

                    if row == range_row[-1]:                # bottom edge
                        row_sup = int(np.ceil(args.patch_size))
                        if column == range_column[0]:        #  bottom left corner
                            column_esq = int(np.floor(args.patch_size))
                        if column == range_column[-1]:       #  bottom right corner
                            column_dir = int(np.floor(args.patch_size))

                    if column == range_column[0]:            # left edge
                        column_esq = int(np.floor(args.patch_size))
                        
                    if column == range_column[-1]:           # right edge
                        column_dir = int(np.floor(args.patch_size))
                        
                    prediction[row-row_inf:row+row_sup,column-column_esq:column+column_dir] = P
        else:
            prediction = model(image.view(1,image.shape[0],image.shape[1],image.shape[2]))


        pbar.finish()
    
    elapsed_time = time.time() - start_time
    prediction = prediction.view(image.shape[1],image.shape[2])
    Output_copy = 1.0 - prediction   # Invert colors
    threshold = 1.0 - args.threshold
    IoU = 0
    ## Loading ground truth (target)
    if args.dataset == 'training':
        os.chdir(rootDir+'/data_semantics/'+str(args.dataset)+'/semantics_binary_ground')
        target = transToTensor(Image.open(img_name,mode='r'))
        IoU = CalculateIoU(target,prediction,threshold=args.threshold,patch_size=args.patch_size,stride=args.stride)
        print('IoU score: {:.3f} %'.format(100*IoU))
    os.chdir(currDir)
    folder = 'results_Sem_Seg_Reconstruction'
    if os.path.isdir(folder):
        os.chdir(folder)
    else:
        os.mkdir(folder)
        os.chdir(folder)
    
    name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    os.mkdir(name)
    os.chdir(os.getcwd()+'/'+name) 
    NAME = 'image_'+str(args.img_number)
    plt.figure(figsize=(12.42,3.75))  
    plt.imshow(Output_copy,cmap=plt.cm.Greys,vmin=0,vmax=1)
    plt.savefig(NAME)
    plt.close()
    # overlapping image
    plt.figure(figsize=(12.42,3.75))  
    plt.imshow(image.permute(1,2,0))
    plt.imshow(Output_copy,cmap='jet',alpha=0.5)
    plt.savefig(NAME+'_overlap')
    plt.close()
    # overlapping image with threshold applied
    Output_copy[Output_copy>threshold]=1.0
    Output_copy[Output_copy<=threshold]=0.0
    plt.figure(figsize=(12.42,3.75))  
    plt.imshow(image.permute(1,2,0))
    plt.imshow(Output_copy,cmap='jet',alpha=0.5)
    plt.savefig(NAME+'_overlap_threshold='+str(args.threshold)+'.png')
    plt.close()
    
    ## Plot the precision-recall curve
    if args.dataset == 'training':    # if the ground-truth is available
        precision, recall, th = precision_recall_curve(target.flatten(),prediction.flatten())
        average_precision = average_precision_score(target.flatten(),prediction.flatten())
        step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Binary Classificaion Precision-Recall Curve: AP={0:0.2f}'.format(average_precision))
        plt.savefig(NAME+'_precision_recall.png')
        plt.close()

    WriteInfoFile2(name,args.patch_size,args.stride,SecondsToText(elapsed_time),args.model_folder,IoU,args.threshold)
    os.chdir(currDir)

if __name__ == '__main__':
    main()