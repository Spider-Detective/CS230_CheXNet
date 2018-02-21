# encoding: utf-8

"""
The main CheXNet model implementation.
"""

import os
import time
import copy
import logging

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import read_data 
import utils

# import customised model, metric and params
import modelSetting.net as net
from evaluate import evaluate

N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
# Training data and entry list
#TRAIN_DATA_DIR = 'images/train'
#TRAIN_IMAGE_LIST = 'train_list.txt'
TRAIN_BATCH_SIZE = 5

# Dev data and entry list
#DEV_DATA_DIR = 'images/dev' 
#DEV_IMAGE_LIST = 'dev_list.txt'
DEV_BATCH_SIZE = 2
use_gpu = torch.cuda.is_available()


normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

# Create the input data pipeline
logging.info("Loading the datasets...")

# a general model definition, scheduler: learning rate decay    
def train_model(model, optimizer, train_loader, loss_fn, metrics, num_epochs=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    #scheduler.step()
    model.train(True)  # Set model to training mode

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        #running_loss = 0.0
        running_corrects = 0
        running_accuracy = 0.0
        loss_avg = utils.RunningAverage()
        # summary for all the mini batch of metrics and loss
        summ = []
        #print(len(train_loader))
        with tqdm(total=len(train_loader)) as t:
            # Iterate over data.
            for (inputs, labels) in train_loader:

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                
                loss = loss_fn(outputs, labels)
 
                loss.backward()

                # performs updates using calculated gradients
                optimizer.step()

                # cutoff by 0.5
                preds = outputs >= 0.5
                preds = preds.type(torch.FloatTensor)

                # Here, we calculate the metrics and the loss for every batch
                # and save them to the summ
                # extract data from torch Variable, move to cpu, convert to
                # numpy
                preds_batch = preds.data.cpu().numpy()
                labels_batch = labels.data.cpu().numpy()

                # Compute all metrics in this batch
                summary_batch = {metric:metrics[metric](preds_batch,labels_batch) for metric in metrics}
                summary_batch['loss'] = loss.data[0]
                summ.append(summary_batch)

                # ToDo, we can use the above summary_batch instead of calculate
                # running_loss for every batch
                loss_avg.update(loss.data[0])

                t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                t.update()

        # Here, when we update all the batch in a certain epoch, we will calculate
        # the mean metrics for this epoch
        # compute mean of all metrics in summary
        metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info("- Train metrics: " + metrics_string)

        if metrics_mean['accuracy'] > best_acc:
            best_acc = metrics_mean['accuracy']
            best_model_wts = copy.deepcopy(model.state_dict())
        
        print("- Train metrics: " + metrics_string)

    print('Best training Acc: {:4f}'.format(best_acc))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model   

# fetch dataloaders
dataloaders = read_data.fetch_dataloader(['train', 'dev'], 'images', 'labels')
train_dl = dataloaders['train']
dev_dl = dataloaders['dev']

# initialize and load the model
model = net.DenseNet121(N_CLASSES)
if use_gpu:
    model = DenseNet121(N_CLASSES).cuda()

criterion = nn.MultiLabelSoftMarginLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)

# Define the metrics
metrics = net.metrics

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model in the training set
model_conv = train_model(model, optimizer, train_dl, criterion, metrics, num_epochs = 4)

utils.save_checkpoint({'state_dict': model.state_dict()}, is_best=None, checkpoint='trial1')
#utils.load_checkpoint(checkpoint = 'trial1/last.pth.tar', model = dev_model)

# evalute the model in the val_dataset
print("Metric Report for the dev set") 
dev_metrics = evaluate(model_conv, criterion, dev_dl, metrics, use_gpu)
