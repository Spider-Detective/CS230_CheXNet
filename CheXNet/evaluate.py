"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import modelSetting.net as net
import read_data
import utils
import sklearn
#from read_data import ChestXrayDataSet
#import model.data_loader as data_loader


# Here, we can add the argument type to support the separate evaluate part
'''
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
'''

def evaluate(model, loss_fn, dataloader, metrics, use_gpu):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()
    sample_size = 0

    print("start evaluate")
    one_but_zero = np.zeros(14)
    zero_but_one = np.zeros(14)
    # summary for current eval loop
    summ = []

    outputs = []
    labels = []
    # compute metrics over the dataset
    for batch_index, (data_batch, labels_batch) in enumerate(dataloader):
        # move to GPU if available
        if use_gpu:
            data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
        
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # converse output probability to prediction data
        output_batch = output_batch >= 0.5
        output_batch = output_batch.type(torch.FloatTensor)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()
        
        outputs.append(output_batch[0])
        labels.append(labels_batch[0])

        sample_size += output_batch.shape[0]

        truth_one_but_zero, truth_zero_but_one = each_label(output_batch, labels_batch)
        one_but_zero += truth_one_but_zero
        zero_but_one += truth_zero_but_one


        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.data[0]
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    
    # calculate the ruc_auc value
    outputs = np.asarray(outputs).astype(int)
    labels = np.asarray(labels).astype(int)
    auc = sklearn.metrics.roc_auc_score(labels,outputs)
    print("ROC_AUC score is :")    
    print(auc)
    # Here is just the screen print out for debug
    print("- Eval metrics : " + metrics_string)


    print(np.array_str(one_but_zero))
    print(np.array_str(zero_but_one))
    return metrics_mean

# ToDo, we can add the separate evaluate part later.

def each_label(outputs, label):
    sample_size = outputs.shape[0]
    prediction = outputs - label
    truth_zero_but_one = np.count_nonzero(prediction == 1, axis = 0)
    truth_one_but_zero = np.count_nonzero(prediction == -1, axis = 0)
    return truth_one_but_zero, truth_zero_but_one


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    #args = parser.parse_args()
    #json_path = os.path.join(args.model_dir, 'params.json')
    #assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    #params = utils.Params(json_path)

    # use GPU if available
    # params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    #if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    #utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))
    #DEV_DATA_DIR = 'images/dev' 
    #DEV_IMAGE_LIST = 'dev_list.txt'
    # Create the input data pipeline
    logging.info("Creating the dataset...")

    use_gpu = torch.cuda.is_available()
    # fetch dataloaders
    dataloaders = read_data.fetch_dataloader(['dev'], 'images', 'labels')
    test_dl = dataloaders['dev']

    logging.info("- done.")

    # Define the model
    # model = net.Net(params).cuda() if params.cuda else net.Net(params)
    # model = utils.load_
    #loss_fn = net.loss_fn
    loss_fn = nn.BCELoss() 
    metrics = net.metrics
    N_CLASSES = 1
    if use_gpu:
        dev_model = net.DenseNet121(N_CLASSES).cuda()
    else:
        dev_model = net.DenseNet121(N_CLASSES)
    utils.load_checkpoint(checkpoint = 'trial1/last.pth.tar', model = dev_model)

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    #utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)
    # Evaluate
    test_metrics = evaluate(dev_model, loss_fn, test_dl, metrics, use_gpu)
    #save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    #utils.save_dict_to_json(test_metrics, save_path)

