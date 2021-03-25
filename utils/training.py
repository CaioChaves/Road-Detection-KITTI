import os
import sys
import math
import string
import random
import shutil
import time 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

from . import imgs as img_utils

RESULTS_PATH = '.results/'
WEIGHTS_PATH = '.weights/'

def save_weights(model, epoch, loss, err):
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            'error': err,
            'state_dict': model.state_dict()
        }, weights_fpath)
    shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.th')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['error']))
    return startEpoch

def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices

def error(preds, targets):
    assert preds.size() == targets.size()
    bs,h,w = preds.size()
    n_pixels = bs*h*w
    incorrect = preds.ne(targets).cpu().sum()
    err = incorrect/n_pixels
    #return round(err,5)
    return err

def CalculateAccuracy(target,prediction,threshold):
    # Class separation for the prediction:
    binary_prediction = torch.zeros_like(prediction)
    binary_prediction[prediction>=threshold] = 1.0

    equal = torch.eq(target,binary_prediction)
    different = 1-equal
    
    TP = int(torch.sum(equal.float()*target))
    TN = int(torch.sum(equal.float()*(1-target)))
    FP = int(torch.sum(different.float()*(1-target)))
    FN = int(torch.sum(different.float()*target))

    total = TP+TN+FP+FN
    Accuracy = (TP+TN)/(total)
    return Accuracy

def train(args, model, trn_loader, optimizer, criterion, device, epoch):
    model.train()
    trn_loss = 0
    trn_error = 0
    for _, data in enumerate(trn_loader):
        inputs = Variable(data[0].to(device))
        targets = Variable(data[1].to(device))

        optimizer.zero_grad()
        output = model(inputs)

        if args.criterion == 'nll':
            targets = targets.view_as(output).long()
        if args.criterion == 'bce':
            targets = targets.view_as(output)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()   
        trn_error += CalculateAccuracy(targets,output,args.threshold)

    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    return trn_loss, trn_error

def test(args, model, test_loader, criterion, device, epoch=1):
    model.eval()
    test_loss = 0
    test_error = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = Variable(data.to(device))
            target = Variable(target.to(device))
            output = model(data)
            
            if args.criterion == 'nll':
                targets = target.view_as(output).long()
            if args.criterion == 'bce':
                targets = target.view_as(output)
            test_loss += criterion(output, targets).item()

            test_error += CalculateAccuracy(targets,output,args.threshold)
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    return test_loss, test_error

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.zero_()

def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    with torch.no_grad():
        for input, target in input_loader:
            data = Variable(input.cuda())
            label = Variable(target.cuda())
            output = model(data)
            pred = get_predictions(output)
            predictions.append([input,target,pred])
    return predictions

def view_sample_predictions(model, loader, n):
    with torch.no_grad():
        inputs, targets = next(iter(loader))
        data = Variable(inputs.cuda())
        label = Variable(targets.cuda())
        output = model(data)
        pred = get_predictions(output)
        batch_size = inputs.size(0)
        for i in range(min(n, batch_size)):
            img_utils.view_image(inputs[i])
            img_utils.view_annotated(targets[i])
            img_utils.view_annotated(pred[i])
