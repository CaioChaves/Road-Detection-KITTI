import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image
import torch
import torchvision
import copy
import math
import time

def SecondsToText(secs):
    days = secs//86400
    hours = (secs - days*86400)//3600
    minutes = (secs - days*86400 - hours*3600)//60
    seconds = secs - days*86400 - hours*3600 - minutes*60
    result = ("{0} day{1}, ".format(days, "s" if days!=1 else "") if days else "") + \
    ("{0} hour{1}, ".format(hours, "s" if hours!=1 else "") if hours else "") + \
    ("{0} minute{1}, ".format(minutes, "s" if minutes!=1 else "") if minutes else "") + \
    ("{0} second{1}, ".format(seconds, "s" if seconds!=1 else "") if seconds else "")
    return result

def PrePlotEvolution(average_tr_loss_pile,average_val_loss_pile,ACC_tr_loss_pile,ACC_val_loss_pile,total_region_rows,total_region_columns):
    if type(average_tr_loss_pile[0])!=list:
        PlotLossEvolution(average_tr_loss_pile,average_val_loss_pile)
        PlotAccuracyEvolution(average_tr_loss_pile,average_val_loss_pile)
    else:
        for j in range(total_region_rows):
            for k in range(total_region_columns):
                PlotLossEvolution(average_tr_loss_pile[j][k],average_val_loss_pile[j][k],region_row=j,region_column=k)                                                   
                PlotAccuracyEvolution(ACC_tr_loss_pile[j][k],ACC_val_loss_pile[j][k],region_row=j,region_column=k)

def PlotLossEvolution(average_tr_loss_pile,average_val_loss_pile,region_row=0,region_column=0):
    plt.figure()
    epochs_x = np.arange(1,len(average_tr_loss_pile)+1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Function')
    plt.title(label='Region: {},{}'.format(region_row,region_column))
    plt.plot(epochs_x,average_tr_loss_pile,'k--',label='Training')
    plt.plot(epochs_x,average_val_loss_pile,'k-.',label='Validation')
    plt.legend(frameon=True)
    plt.savefig('loss_function_graph_region_{}_{}'.format(region_row,region_column))
    plt.close()

def PlotAccuracyEvolution(train,validation,region_row=0,region_column=0):
    plt.figure()
    epochs_x = np.arange(1,len(train)+1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(label='Region: {},{}'.format(region_row,region_column))
    plt.plot(epochs_x,train,'k--',label='Training')
    plt.plot(epochs_x,validation,'k-.',label='Validation')
    plt.legend(frameon=True)
    plt.savefig('accuracy_graph_region_{}_{}'.format(region_row,region_column))
    plt.close()
    
def WriteInfoFile(name,patch_size,ppi,ppi_coeff,data_split,batch_size,epochs,optimizer,elapsed_time,loss_function,network,num_regions=1,model_param=0):
    file = open('model_info.txt','w+')
    file.write('Date and hour the test ended: '+name+'\n')
    file.write('\nPatch size: '+str(patch_size)+'x'+str(patch_size)+'\n')
    file.write('\nPatchs randomly picked per image: '+str(ppi)+'\n')
    file.write('\nPatchs per image coefficient: '+str(ppi_coeff)+'\n')
    file.write('\nFraction of data for train dataset: '+str(data_split)+'\n')
    file.write('\nBatch size: '+str(batch_size)+'\n')
    file.write('\nEpochs: '+str(epochs)+'\n')    
    file.write('\nOptimizer State: '+str(optimizer)+'\n')
    file.write('\nTotal elapsed time (training included): '+str(elapsed_time)+'\n')
    file.write('\nNumber of regions: '+str(num_regions)+'\n')
    file.write('\nLoss Function: '+str(loss_function)+'\n')
    file.write('\nNetwork architecture: '+str(network)+'\n')
    file.write('\nNumber of trainable parameters in the model: '+str(model_param)+'\n')
    file.close()

def WriteInfoFile2(name,patch_size,stride,elapsed_time,model_name,iou_score,threshold):
    file = open('model_info.txt','w+')
    file.write('Date and hour the test ended: '+name+'\n')
    file.write('\nPatch size: '+str(patch_size)+'x'+str(patch_size)+'\n')
    file.write('\nStride value: '+str(stride)+'\n')
    file.write('\nTotal elapsed time: '+str(elapsed_time)+'\n')
    file.write('\nPre-trained network used: '+str(model_name)+'\n')
    file.write('\nIoU score: '+str(iou_score)+'\n')
    file.write('\nThreshold: '+str(threshold)+'\n')
    file.close()

def WriteInfoFile3(name,elapsed_time,loss_function, optimizer, network, pre_trained, model_param):
    file = open('model_info.txt','w+')
    file.write('Date and hour the test ended: '+name+'\n')
    file.write('\nTotal elapsed time (training included): '+str(elapsed_time)+'\n')
    file.write('\nLoss Function: '+str(loss_function)+'\n')
    file.write('\nOptimizer State: '+str(optimizer)+'\n')
    file.write('\nNetwork architecture: '+str(network)+'\n')
    file.write('\nPre-trained torchvision model: '+str(pre_trained)+'\n')
    file.write('\nNumber of trainable parameters in the model: '+str(model_param)+'\n')
    file.close()

def AccuracyTest(output,target):
    increment = 0
    pos_cm = [] 
    batch_size = len(output)
    for i in range(batch_size):
        if output[i] >= 0.5 and target[i] >= 0.5:
            pos_cm.append([0,0])      # predicted ground / truth ground
            increment += 1  
        elif output[i] <= 0.5 and target[i] < 0.5:
            pos_cm.append([1,1])      # predicted non-ground / truth non-ground
            increment += 1  
        elif output[i] >= 0.5 and target[i] < 0.5:
            pos_cm.append([0,1])      # predicted ground / truth non-ground
            increment += 0
        elif output[i] <= 0.5 and target[i] >= 0.5:
            pos_cm.append([1,0])     # predicted non-ground / truth non-ground
            increment += 0
        else:
            print("Accuracy_test Error!")
    return pos_cm, increment

def PlotConfusionMatrix(epoch,confusion_matrix,normalize=False,title=None,cmap=plt.cm.Greys,phase=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix - '+phase+' - Epoch: '+str(epoch+1)
        else:
            title = 'Confusion matrix, without normalization - '+phase+' - Epoch: '+str(epoch+1)
    cm = confusion_matrix
    if normalize:
        RowSum = np.transpose(np.reshape(np.sum(cm,axis=1),(1,2)))
        cm = cm/RowSum 
    classes = ['Navigable','Non-navigable']

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    im.set_clim(0,1)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else '.1f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    NAME = str(epoch+1)+'_epoch_confusion_matrix_'+str(phase)
    fig.savefig(NAME)
    plt.close(fig)

def WriteCSVFile(epochs,loss_func_tr,loss_func_val,accuracy_tr,accuracy_val):
    with open('LearningStatistics.csv','w') as csv_file:
        csv_writer = csv.writer(csv_file,delimiter='\t')
        csv_writer.writerow(['epoch','loss_func_tr','loss_func_val','accuracy_tr','accuracy_val'])
        for i in range(epochs):
            csv_writer.writerow([str(i+1),str(loss_func_tr[i]),str(loss_func_val[i]),str(accuracy_tr[i]),str(accuracy_val[i])])
    csv_file.close()

def WriteCSVFile2(epochs,loss_func_tr,loss_func_val,accuracy_tr,accuracy_val,total_region_rows,total_region_columns):
    with open('LearningStatistics.csv','w') as csv_file:
        csv_writer = csv.writer(csv_file,delimiter='\t')
        header = ['epochs']
        for region_row in range(total_region_rows):
            for region_column in range(total_region_columns):
                header.append('loss_func_tr_region_{}_{}'.format(region_row,region_column))
                header.append('loss_func_val_region_{}_{}'.format(region_row,region_column))
                header.append('accuracy_tr_region_{}_{}'.format(region_row,region_column))
                header.append('accuracy_val_region_{}_{}'.format(region_row,region_column))
        csv_writer.writerow(header)
        aux = []
        for i in range(epochs):
            aux.append(str(i+1))
            for region_row in range(total_region_rows):
                for region_column in range(total_region_columns):
                        aux.append(str(loss_func_tr[region_row][region_column][i]))
                        aux.append(str(loss_func_val[region_row][region_column][i]))
                        aux.append(str(accuracy_tr[region_row][region_column][i]))
                        aux.append(str(accuracy_val[region_row][region_column][i]))
            csv_writer.writerow(aux)
            aux = []
    csv_file.close()

def GetCropLimits(region,division,img_size):
    """
    Arguments: region: [row,column] of the current region
               division: [number of rows, number of columns]
               original image has shape 3x370x1224
    """
    R = np.linspace(0,img_size[0],division[0]+1)
    C = np.linspace(0,img_size[1],division[1]+1)
    Rlim = [int(round(R[region[0]])),int(round(R[region[0]+1]))]
    Clim = [int(round(C[region[1]])),int(round(C[region[1]+1]))]
    print('Region [{},{}] / Row limits: {} / Column limits: {}'.format(region[0],region[1],Rlim,Clim))
    return Rlim, Clim

def CalculateIoU(target,prediction,threshold,patch_size=28,stride=1):
    # Class separation for the prediction:
    binary_prediction = torch.zeros_like(prediction)
    binary_prediction[prediction>=threshold] = 1.0
    intersection = torch.sum(torch.eq(target,binary_prediction))
    union = torch.numel(binary_prediction)
    iou_score = intersection.item()/union
    return iou_score

def CalculateMetrics(target,prediction,threshold):
    # Class separation for the prediction:
    binary_prediction = torch.zeros_like(prediction)
    binary_prediction[prediction>=threshold] = 1.0

    intersection = torch.sum(torch.eq(target,binary_prediction))
    union = torch.numel(binary_prediction)
    iou_score = intersection.item()/union

    equal = torch.eq(target,binary_prediction)
    different = 1-equal
    
    TP = int(torch.sum(equal.float()*target))
    TN = int(torch.sum(equal.float()*(1-target)))
    FP = int(torch.sum(different.float()*(1-target)))
    FN = int(torch.sum(different.float()*target))

    total = TP+TN+FP+FN
    Dice = 2*TP/(FP+FN+2*TP) if FP+FN+2*TP > 0 else 0
    Accuracy = (TP+TN)/(total)
    Precision = (TP)/(TP+FP) if TP+FP > 0 else 0
    Recall = (TP)/(TP+FN) if TP+FN > 0 else 0
    return iou_score,TP,TN,FP,FN,total,Dice,Accuracy,Precision,Recall

def Convert2PolarCoordinates(patch_y_pos,patch_x_pos,heigth,width,normalized=True):
    D = math.sqrt(math.pow(heigth,2)+math.pow(width/2,2)) if normalized else 1     #Coefficient for distance normalization
    pho = math.sqrt(math.pow(patch_y_pos-heigth,2)+math.pow(patch_x_pos-width/2,2))/D
    theta = math.atan2(heigth-patch_y_pos,patch_x_pos-width/2)/3.14
    return pho,theta

def Convert2CenteredCoordinates(patch_y_pos,patch_x_pos,heigth,width):
    row = (patch_y_pos-heigth/2)/heigth
    column = (patch_x_pos-width/2)/heigth
    return row,column