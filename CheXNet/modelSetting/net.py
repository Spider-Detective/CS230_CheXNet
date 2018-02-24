"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# We can define our own custom model here and import into the our trainning
# function

class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x 

# Here is our own customized defined loss function, we can made our customed loss
# function here.


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples


# Here is our customized
def accuracy(outputs, label):
    np.savetxt('output.csv', outputs - label, delimiter = " ", fmt = '%1d')
    return (1 - np.count_nonzero(np.linalg.norm(outputs - label, axis = 1)) / outputs.shape[0]) 

def total_accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size * 14 - 14 means the size our
        output vecor (14 diseases)
        labels: (np.ndarray) dimension batch_size * 14

    Returns: (float) accuracy in [0,1]
    """
    return np.sum(outputs==labels)/float(labels.size) 

def ROC_AUC(outputs,labels):
    return roc_auc_score(labels, outputs)

def precision(outputs,labels):
    return precision_score(labels, outputs, average='weighted')

def recall(outputs,labels):
    return recall_score(labels, outputs, average='weighted')

def f1(outputs,labels):
    return f1_score(labels, outputs, average='weighted')
    
# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    #'exact_accuracy': overall_accuracy,
    # could add more metrics such as accuracy for each token type
    'accuracy': accuracy,
    'total_accuracy': total_accuracy,
    'ROC_AUC': ROC_AUC,
    # 'precision':precision,
    # 'recall':recall,
    # 'f1':f1
}
