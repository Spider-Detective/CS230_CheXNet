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
N_CLASSES = 15
def evaluate(encoder, decoder, dataloader, metrics, loss_fn, use_gpu, logger):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    encoder.eval()
    decoder.eval()

    false_positive = [0] * N_CLASSES
    false_negative = [0] * N_CLASSES
    # summary for current eval loop
    summ = []
    outputs_batch = []
    labels_batch = []

    loss_avg = utils.RunningAverage()
    
    # add tqdm 
    with tqdm(total = len(dataloader)) as t:
        # compute metrics over the dataset
        for data, labels in dataloader:

            # move to GPU if available
            if use_gpu:
                data, labels = data.cuda(async=True), labels.cuda(async=True)

            # fetch the next evaluation batch
            data, labels = Variable(data), Variable(labels)

            # compute model output
            pred_features = encoder(data)
            outputs = decoder(pred_features)

            loss = loss_fn.compute(outputs, labels)
            loss_avg.update(loss.data[0])
            
            # reshape into 1D numpy array and output for all batches
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            outputs = outputs.data.cpu().numpy().reshape((1,N_CLASSES))
            labels = labels.data.cpu().numpy().reshape((1,N_CLASSES))
            
            # save for auc calculation
            if (len(outputs_batch) == 0):  # first output and label
                outputs_batch = outputs
                labels_batch = labels
            else:
                outputs_batch = np.vstack((outputs_batch, outputs))
                labels_batch = np.vstack((labels_batch, labels))

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](outputs, labels)
                             for metric in metrics}
            summary_batch['loss'] = loss.data[0]
            summ.append(summary_batch)

            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.6f}".format(k, v) for k, v in metrics_mean.items())
    logger.info("- Eval metrics : " + metrics_string)

    auc = net.computeROC_AUC(outputs_batch, labels_batch) 
    logger.info("ROC AUC is :")
    logger.info(auc)

    metrics_mean['auc_mean'] = np.mean(auc)
    logger.info("Mean value of AUC: ")
    logger.info(metrics_mean['auc_mean'])
    
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
    logger.info("Creating the dataset...")
    
    # check gpu
    use_gpu = torch.cuda.is_available()
    
    # fetch dataloaders
    if use_gpu:
        dataloaders = read_data.fetch_dataloader(['dev'], '/home/ubuntu/Data_Processed/images', '/home/ubuntu/Data_Processed/labels')
    else:
        dataloaders = read_data.fetch_dataloader(['dev'], 'images', 'labels')
    
    dev_dl = dataloaders['dev']

    logger.info("- done.")

    # Define the model
    # model = net.Net(params).cuda() if params.cuda else net.Net(params)
    loss_fn = net.MultiLabelLoss()
    metrics = net.metrics

    embed_size = 50
    hidden_size = 100
    num_layers = 2
    encoder = net.DenseNet121(embed_size)
    decoder = net.DecoderRNN(embed_size, hidden_size, N_CLASSES, num_layers)

    if use_gpu:
        encoder = encoder.cuda()
        encoder = torch.nn.DataParallel(encoder)

        decoder = decoder.cuda()
        decoder = torch.nn.DataParallel(decoder)

    #utils.load_checkpoint(checkpoint = 'trial1/last.pth.tar', model = dev_model)

    logger.info("Starting evaluation")

    # Reload weights from the saved file
    #utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    dev_metrics, dev_loss = evaluate(encoder, decoder, dev_dl, metrics, loss_fn, use_gpu)

    #save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    #utils.save_dict_to_json(test_metrics, save_path)

