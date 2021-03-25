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
import torch.utils.tensorboard as TensorBoard 
from patchCNN_tools import WriteInfoFile2, SecondsToText, CalculateIoU, Convert2CenteredCoordinates, CalculateMetrics
from PIL import Image
from progressbar import ProgressBar, Percentage, Bar, ETA, AdaptiveETA
from sklearn.metrics import precision_recall_curve, average_precision_score
from inspect import signature
import csv
import cv2
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
    parser.add_argument('--input-video', type=str, 
                        help='input video folder')
    parser.add_argument('--patch-size', type=int, default=28, metavar='N',
                        help='input patch size for crooping image (default: 28)')
    parser.add_argument('--stride', type=int, default=10,
                        help='step value between sucessive patchs (default: 10)')
    parser.add_argument('--epoch-number', type=int, 
                        help='input epoch number (default=none)')
    parser.add_argument('--model-folder',type=str,dest='model_folder',
                        help='Folder name containing model (.pt)')
    parser.add_argument('--threshold', type=float, default=0.5, metavar='N',
                        help='threshold value for class separation between 0.0 and 1.0 (default: 0.5)')
    parser.add_argument('--arch', type=str,dest='model_arch',default='base',
                        help='string to identify which CNN architecture to use (options: base (default), mini, MS)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dataset', type=str, default='training',
                        help='String indicating which folder to pick image from (options: testing, training / default: training)')
    parser.add_argument('--spatial-feats', action='store_true', default=False,
                        help='Include position features to CNN (def. False)')
    parser.add_argument('--txt', type=int, default=1,
                        help='depois joga isso fora')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4} if use_cuda else {}
    currDir = os.getcwd()
  
    # Load model
    if args.model_arch == 'base':
        model = MyNetwork()
        model.load_state_dict(torch.load(currDir+'/results/'+args.model_folder+'/patch_cnn_models/epoch_{}_patch_cnn_model.pth'.format(args.epoch_number)))
    if args.model_arch == 'mini':
        model = MinimalisticCNN(spatial_features=args.spatial_feats)
        model.load_state_dict(torch.load(currDir+'/results/'+args.model_folder+'/patch_cnn_models/epoch_{}_patch_cnn_model.pth'.format(args.epoch_number)))
    if args.model_arch == 'MS':
        model = MultiScalePatchNet(spatial_features=args.spatial_feats)
        model.load_state_dict(torch.load(currDir+'/results/'+args.model_folder+'/patch_cnn_models/epoch_{}_patch_cnn_model.pth'.format(args.epoch_number)))
    if args.model_arch == 'tiramisu':
        model = tiramisu.FCDenseNet57(n_classes=1,outfunc='sigmoid')
        model.load_state_dict(torch.load(currDir+'/resultsTiramisu/'+args.model_folder+'/patch_cnn_models/epoch_{}_patch_cnn_model.pth'.format(args.epoch_number)))
   
    model.to(device)
    model.eval()

    ## Validation images
    #validation_indices = [int(line.strip()) for line in open('validation_indices.txt')]
    if args.txt == 1:
        images_indices = [int(line.strip()) for line in open('remaining_indices.txt')]
    if args.txt == 2:
        images_indices = [int(line.strip()) for line in open('images_video_indices.txt')]
    if args.txt == 3:
        images_indices = [int(line.strip()) for line in open('images_video_indices_05.txt')]

    widgets = [Percentage(),' ',Bar(),' ',AdaptiveETA()]
    pbar = ProgressBar(widgets=widgets)
    image_shape = [3,512,1392]   # Channels x rows x column
    range_row = range(int(args.patch_size/1),image_shape[1]-int(args.patch_size/1),args.stride)
    range_column = range(int(args.patch_size/1),image_shape[2]-int(args.patch_size/1),args.stride)
    rootDir = os.getcwd()
    folder = 'results_Sem_Seg_Reconstruction_video'
    name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')+'_'+str(args.model_arch)+'_pos_'+str(args.spatial_feats)+'_stride_'+str(args.stride)
    transToTensor = torchvision.transforms.ToTensor()
    transToPILImage = torchvision.transforms.ToPILImage()
    Final_images_list = []
    Final_images_threhsold_list = []
    

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
 
    if os.path.isdir(folder):
        os.chdir(folder)
    else:
        os.mkdir(folder)
        os.chdir(folder)
                    
    if os.path.isdir(name):
        os.chdir(name)
    else:
        os.mkdir(name)
        os.chdir(name)

    os.chdir(rootDir+'/'+folder+'/'+name)
    out = cv2.VideoWriter('video_threshold.avi',fourcc, 10.0, (1242,375))
    out2 = cv2.VideoWriter('video.avi',fourcc, 10.0, (1242,375))

    start_time = time.time()

    os.chdir(rootDir)
    with torch.no_grad():
        for img_number in images_indices:
            if 0 <= img_number <= 9:
                    img_name = '000000000'+str(img_number)+'.png'
            elif 10 <= img_number <= 99:
                    img_name = '00000000'+str(img_number)+'.png'
            elif 100 <= img_number <=999:
                    img_name = '0000000'+str(img_number)+'.png'
                
            ## Loading image
            os.chdir(rootDir+'/2011_09_26/'+args.input_video+'/image_03/data')
            image = transToTensor(Image.open(img_name,mode='r'))
            prediction = torch.zeros(image.shape[1],image.shape[2])
            os.chdir(rootDir)
            # Loop Scanning
            
            pbar.start()
            if args.model_arch != 'tiramisu':
                for row in pbar(range_row):
                    for column in range_column:
                        small_patch = image[:,row-int(args.patch_size/4):row+int(args.patch_size/4),column-int(args.patch_size/4):column+int(args.patch_size/4)]
                        small_patch = small_patch.view(1,3,int(args.patch_size/2),int(args.patch_size/2))

                        medium_patch = image[:,row-int(args.patch_size/2):row+int(args.patch_size/2),column-int(args.patch_size/2):column+int(args.patch_size/2)]
                        medium_patch = medium_patch.view(1,3,args.patch_size,args.patch_size)

                        large_patch = image[:,row-int(args.patch_size/1):row+int(args.patch_size/1),column-int(args.patch_size/1):column+int(args.patch_size/1)]
                        large_patch = large_patch.view(1,3,int(args.patch_size*2),int(args.patch_size*2))
                        
                        r, c = Convert2CenteredCoordinates(row,column,heigth=image.shape[1],width=image.shape[2])       ## Spatial - features
                    
                        if args.model_arch == 'base':
                            P = model(medium_patch.to(device))
                        if args.model_arch == 'mini':
                            P = model(medium_patch.to(device),device,r,c)
                        if args.model_arch == 'MS':
                            P = model(small_patch.to(device),medium_patch.to(device),large_patch.to(device),device,r,c)
                        
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
                prediction = prediction.view(image.shape[1],image.shape[2])


            pbar.finish()
            
            Output_copy = 1.0 - prediction   # Invert Colors
                           
            os.chdir(rootDir+'/'+folder+'/'+name)
            NAME = 'image_'+str(img_number)
            print(NAME)
            plt.figure(figsize=(12.42,3.75))
            plt.imshow(Output_copy,cmap=plt.cm.Greys,vmin=0,vmax=1)
            plt.savefig(NAME)
            plt.close()
            # overlapping image
            plt.figure(figsize=(12.42,3.75))
            plt.imshow(image.permute(1,2,0))
            plt.imshow(Output_copy,cmap='jet',alpha=0.5)
            plt.savefig(NAME+'_overlap.png')
            plt.close()

            img = cv2.imread(NAME+'_overlap.png',1)
            out2.write(img)
                                    
            threshold = round(args.threshold,2)
            # overlapping image with threshold applied
            aux_output = torch.zeros_like(prediction)
            aux_output[prediction>threshold]=1.0
            aux_output[prediction<=threshold]=0.0
            plt.figure(figsize=(12.42,3.75))
            plt.imshow(image.permute(1,2,0))
            plt.imshow(1.0-aux_output,cmap='jet',alpha=0.5)  
            plt.savefig(NAME+'_overlap_threshold='+str(threshold)+'_stride_'+str(args.stride)+'.png')
            plt.close()

            img = cv2.imread(NAME+'_overlap_threshold='+str(threshold)+'_stride_'+str(args.stride)+'.png',1)
            out.write(img)

            os.chdir(rootDir+'/'+folder+'/'+name)

                
    
    #end loop over all images
    out.release()
    out2.release()
    elapsed_time = time.time() - start_time              
    WriteInfoFile2(name,args.patch_size,args.stride,SecondsToText(elapsed_time),args.model_folder,0,args.threshold)
    os.chdir(rootDir)


if __name__ == '__main__':
    main()

