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
import torch.optim.lr_scheduler as lr_scheduler
# import customised model, metric and params
import modelSetting.net as net
from evaluate import evaluate


N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

# Training data and entry list
#TRAIN_BATCH_SIZE = 5

# Dev data and entry list
#DEV_BATCH_SIZE = 1

# Check if GPU is available on current platform
use_gpu = torch.cuda.is_available()

normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

utils.set_logger(os.path.join(os.getcwd(),'train.log'))
logging.info("Loading the datasets...")

# a general model definition, scheduler: learning rate decay    
def train(model, optimizer, scheduler, train_loader, loss_fn, metrics):

    #scheduler.step()
    model.train(True)  # Set model to training mode

    running_accuracy = 0.0
    loss_avg = utils.RunningAverage()

    false_positive = [0] * N_CLASSES
    false_negative = [0] * N_CLASSES

    # summary for all the mini batch of metrics and loss
    summ = []

    with tqdm(total=len(train_loader)) as t:
        # Iterate over data.
         for data in train_loader:
            # get the inputs
            inputs, labels = data

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

            # extract data from torch Variable, move to cpu, convert to numpy
            preds_batch = preds.data.cpu().numpy()
            labels_batch = labels.data.cpu().numpy()

            # Compute all metrics in this batch
            summary_batch = {metric:metrics[metric](preds_batch,labels_batch) for metric in metrics}
            summary_batch['loss'] = loss.data[0]
            summ.append(summary_batch)

            false_positive_batch, false_negative_batch = net.compare_pred_and_label(preds_batch, labels_batch)
            false_positive += false_positive_batch
            false_negative += false_negative_batch
           
            loss_avg.update(loss.data[0])

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # Here, when we update all the batch in a certain epoch, we will calculate
    # the mean metrics for this epoch, compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    
    #logging.info("- Train metrics: " + metrics_string)
    
    logging.info("- Train metrics: %s", metrics_string)
    logging.info("False positives of each disease: %s", np.array_str(false_positive))
    logging.info("False negatives of each disease: %s", np.array_str(false_negative))
    
    # model.load_state_dict(best_model_wts)


def train_and_evaluate(model, optimizer, scheduler, train_loader, dev_loader, loss_fn, metrics, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.0

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)
        train(model, optimizer, scheduler, train_loader, loss_fn, metrics)
        logging.info("\n")

        # evalute the model in the dev_dataset
        logging.info("Metric Report for the dev set") 
        dev_metrics, dev_loss = evaluate(model, dev_loader, metrics, loss_fn, use_gpu)
        scheduler.step(dev_loss)
        #dev_auc = dev_metrics['auc_mean']
        if dev_loss > best_loss:
            logging.info("Found better model!")
            best_loss = dev_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        logging.info('\n')
    # logging.info report
    logging.info('Best eval loss: {:4f}'.format(best_loss))
    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
           time_elapsed // 60, time_elapsed % 60))
    # load best model weights
    model.load_state_dict(best_model_wts)


# Set the random seed for reproducible experiments
torch.manual_seed(230)
if use_gpu: torch.cuda.manual_seed(230)

# fetch dataloaders
if use_gpu:
    dataloaders = read_data.fetch_dataloader(['train', 'dev'], '/home/ubuntu/Data_Processed/images', '/home/ubuntu/Data_Processed/labels')
else:
    dataloaders = read_data.fetch_dataloader(['train', 'dev'], 'images', 'labels')
   
train_dl = dataloaders['train']
dev_dl = dataloaders['dev']

# initialize and load the model
model = net.DenseNet121(N_CLASSES)
if use_gpu:
    model = net.DenseNet121(N_CLASSES).cuda()
    model = torch.nn.DataParallel(model)


#weights_file = os.path.join('/home/ubuntu/Data_Processed/labels/','train_list.txt')
#train_weight = torch.from_numpy(utils.get_loss_weights(weights_file)).float()
#logging.info(train_weight)
#if use_gpu:
#   train_weight = train_weight.cuda()

#train_loss = nn.MultiLabelSoftMarginLoss(weight = train_weight) 
train_loss = nn.MultiLabelSoftMarginLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)

# Define the metrics
metrics = net.metrics
# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
plat_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,patience = 1)

# Train the model in the training set
logging.info("Names of 14 diseases:")
#[logging.info('Type={}'.format(i),'Disease={}'.format(name)) for i, name in enumerate(CLASS_NAMES)]
train_and_evaluate(model, optimizer, plat_lr_scheduler, train_dl, dev_dl, train_loss, metrics,num_epochs = 5)
utils.save_checkpoint({'state_dict': model.state_dict()}, is_best=None, checkpoint='trial1')
#utils.load_checkpoint(checkpoint = 'trial1/last.pth.tar', model = dev_model)


