

"""
The main CheXNet model implementation.
"""

import os
import time
import copy
import logging
import sys

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


N_CLASSES = 15
CLASS_NAMES = [ 'No finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


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
            loss = loss_fn.compute(outputs, labels)
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

            #false_positive_batch, false_negative_batch = net.compare_pred_and_label(preds_batch, labels_batch)
            #false_positive += false_positive_batch
            #false_negative += false_negative_batch
           
            loss_avg.update(loss.data[0])

            t.set_postfix(loss='{:05.6f}'.format(loss_avg()))
            t.update()

    # Here, when we update all the batch in a certain epoch, we will calculate
    # the mean metrics for this epoch, compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.6f}".format(k, v) for k, v in metrics_mean.items())
    
    logger.info("- Train metrics: %s", metrics_string)
    #logger.info("False positives of each disease: %s", np.array_str(false_positive))
    #logger.info("False negatives of each disease: %s", np.array_str(false_negative))

def train_and_evaluate(model, optimizer, scheduler, train_loader, dev_loader, loss_fn, metrics, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = sys.float_info.max

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)
        train(model, optimizer, scheduler, train_loader, loss_fn, metrics)
        logger.info("\n")

        # evalute the model in the dev_dataset
        logger.info("Metric Report for the dev set") 
        dev_metrics, dev_loss = evaluate(model, dev_loader, metrics, loss_fn,\
                                         use_gpu, logger)
        scheduler.step(dev_loss)

        # find the best model based on the dev loss
        if dev_loss < best_loss:
            logger.info("Found better model!")
            best_loss = dev_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        logger.info('\n')
    # logger.info report
    logger.info('Best eval loss: {:4f}'.format(best_loss))
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
           time_elapsed // 60, time_elapsed % 60))
    # load best model weights
    model.load_state_dict(best_model_wts)

# construct options
model_option = {"densenet" : net.DenseNet121(N_CLASSES) ,
                "resnet" : net.ResNet18(N_CLASSES)}

lr_option = {"1e-3" : 1e-3 , "5e-4" : 5e-4 ,"1e-4" : 1e-4 ,"5e-5" : 5e-5}

loss_option = {"loss_1" : net.MultiLabelLoss() , "loss_2" : net.MultiLabelLoss2()}

# Check if GPU is available on current platform
use_gpu = torch.cuda.is_available()


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


for model_name , model in model_option.items():
    for lr_name, lr_value in lr_option.items():
        for loss_name, train_loss in loss_option.items():
            # save the output into a log file
            filename ="train_" + model_name + lr_name + loss_name 
            utils.set_logger(filename ,os.path.join(os.getcwd(),"logFiles/" + filename + ".log"))
            logger = logging.getLogger(filename)
            logger.info("Loading the datasets...")
            if use_gpu:
                model = model.cuda()
                model = torch.nn.DataParallel(model)

            #weights_file = os.path.join('/home/ubuntu/Data_Processed/labels/','train_list.txt')
            #train_weight = torch.from_numpy(utils.get_loss_weights(weights_file)).float()
            #logger.info(train_weight)
            #if use_gpu:
            #   train_weight = train_weight.cuda()

            #train_loss = nn.MultiLabelSoftMarginLoss(weight = train_weight) 
            #train_loss = net.MultiLabelLoss2()
            optimizer = optim.Adam(model.parameters(), lr=lr_value, weight_decay=5e-5)

            # Define the metrics
            metrics = net.metrics
            # Decay LR by a factor of 0.1 every 7 epochs
            # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
            plat_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = 1, verbose=True, threshold=0.01)

            # Train the model in the training set
            logger.info("Names of 14 diseases:")
            #[logger.info('Type={}'.format(i),'Disease={}'.format(name)) for i, name in enumerate(CLASS_NAMES)]
            train_and_evaluate(model, optimizer, plat_lr_scheduler, train_dl, dev_dl, train_loss, metrics,num_epochs = 1)
            utils.save_checkpoint({'state_dict': model.state_dict()}, is_best=None, \
                                  checkpoint= "./checkPoint/" + filename)
            #utils.load_checkpoint(checkpoint = 'trial1/last.pth.tar', model = dev_model)


