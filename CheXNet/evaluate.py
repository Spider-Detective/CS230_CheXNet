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
from tqdm import tqdm
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
N_CLASSES = 14
def evaluate(model, dataloader, metrics, loss_fn, use_gpu):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    false_positive = [0] * N_CLASSES
    false_negative = [0] * N_CLASSES
    # summary for current eval loop
    summ = []
    preds = []
    labels = []

    loss_avg = utils.RunningAverage()
    
    # add tqdm 
    with tqdm(total = len(dataloader)) as t:
        # compute metrics over the dataset
        for data_batch, labels_batch in dataloader:

            # move to GPU if available
            if use_gpu:
                data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)

            # fetch the next evaluation batch
            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

            # compute model output
            preds_batch = model(data_batch)
            loss = loss_fn.compute(preds_batch, labels_batch)
            loss_avg.update(loss.data[0])
            
            # reshape into 1D numpy array and output for all batches
            #preds.append(preds_batch.data.cpu().numpy().reshape(14))
            #labels.append(labels_batch.data.cpu().numpy().reshape(14))

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            preds_batch = preds_batch.data.cpu().numpy().reshape((1,14))
            labels_batch = labels_batch.data.cpu().numpy().reshape((1,14))

            # save for auc calculation
            if (len(preds) == 0):
                preds = preds_batch
                labels = labels_batch
            else:
                preds = np.vstack((preds, preds_batch))
                labels = np.vstack((labels, labels_batch))

            # converse output probability to prediction data
            #preds_batch = preds_batch >= 0.5
            #preds_batch = preds_batch.type(torch.FloatTensor)

            #false_positive_batch, false_negative_batch = net.compare_pred_and_label(preds_batch, labels_batch)
            #false_positive += false_positive_batch
            #false_negative += false_negative_batch

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](preds_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.data[0]
            summ.append(summary_batch)

            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    auc = net.computeROC_AUC(preds, labels) 
    logging.info("ROC AUC is :")
    logging.info(auc)
    metrics_mean['auc_mean'] = np.mean(auc)
    
    return metrics_mean, loss_avg()


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
    #params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    #if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    #utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    
    # check gpu
    use_gpu = torch.cuda.is_available()
    
    # fetch dataloaders
    if use_gpu:
        dataloaders = read_data.fetch_dataloader(['dev'], '/home/ubuntu/Data_Processed/images', '/home/ubuntu/Data_Processed/labels')
    else:
        dataloaders = read_data.fetch_dataloader(['dev'], 'images', 'labels')
    
    dev_dl = dataloaders['dev']

    logging.info("- done.")

    # Define the model
    # model = net.Net(params).cuda() if params.cuda else net.Net(params)
    loss_fn = net.MultiLabelLoss()
    metrics = net.metrics

    dev_model = net.DenseNet121(N_CLASSES)
    if use_gpu:
        dev_model = net.DenseNet121(N_CLASSES).cuda()
        dev_model = torch.nn.DataParallel(dev_model)

    #utils.load_checkpoint(checkpoint = 'trial1/last.pth.tar', model = dev_model)

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    #utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    dev_metrics, dev_loss = evaluate(dev_model, dev_dl, metrics, loss_fn, use_gpu)

    #save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    #utils.save_dict_to_json(test_metrics, save_path)

