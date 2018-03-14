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
from torch.autograd import Variable
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

class ResNet18(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x 
def compare_pred_and_label(outputs, labels):
    '''compare the prediciton with true labels, and return the number of false positives and negatives'''
    difference = outputs - labels
    false_positive = np.count_nonzero(difference == 1, axis = 0)  # 1 - 0 
    false_negative = np.count_nonzero(difference == -1, axis = 0) # 0 - 1
    return false_positive, false_negative

# Here is our customized
def accuracy(outputs, labels):
    return (1 - np.count_nonzero(np.linalg.norm(outputs - labels, axis = 1)) / outputs.shape[0]) 

# Here is our customized
def dev_accuracy(outputs, labels):
    #np.savetxt('output.csv', outputs - labels, delimiter = " ", fmt = '%1d')
    return (1 - np.count_nonzero(np.linalg.norm(outputs - labels, axis = 0)) / outputs.shape[0]) 

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

def computeROC_AUC(outputs, labels):
    auc = []
    # Calculate AUC, catch the error if not possible to calculate
    for i in range(0,15):
        try:
            single_auc = roc_auc_score(labels[:,i],outputs[:,i])
        except ValueError:
            print("AUC value error! Cannot calculate AUC.")
            single_auc = 0           
        auc.append(single_auc) 
    return auc
    
# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    #'accuracy': accuracy,
    'total_accuracy': total_accuracy,
    #'ROC_AUC': ROC_AUC,
    # 'precision':precision,
    # 'recall':recall,
    # 'f1':f1
}

class MultiLabelLoss2():
    """Creates a criterion that optimizes a multi-label one-versus-all
    loss based on max-entropy, between input `x` and target `y` of size `(N, C)`.
    For each sample in the minibatch::

       loss(x, y) = - sum_i (y[i] * log( 1 / (1 + exp(-x[i])) )
                         + ( (1-y[i]) * log(exp(-x[i]) / (1 + exp(-x[i])) ) )

    where `i == 0` to `x.nElement()-1`, `y[i]  in {0,1}`.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
           class. If given, it has to be a Tensor of size `C`. Otherwise, it is
           treated as if having all ones.
        size_average (bool, optional): By default, the losses are averaged over
            observations for each minibatch. However, if the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch.
            Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on :attr:`size_average`. When
            :attr:`reduce` is ``False``, returns a loss per batch element instead and
            ignores :attr:`size_average`. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` where `N` is the batch size and `C` is the number of classes.
        - Target: :math:`(N, C)`, same shape as the input.
        - Output: scalar. If `reduce` is False, then `(N)`.
    """

    def compute(self, input, target):
        classes = input.size()[1]
        num_example = input.size()[0]
        #loss = target[:,0] * F.binary_cross_entropy(input, target, weight=None, size_average=True)
        #loss = loss + (1 - target[]) * ()
        
        use_gpu = torch.cuda.is_available()
        # calculate the 14 disease cross_entropy
        if use_gpu:
            loss = Variable(torch.zeros(num_example,1).type(torch.cuda.FloatTensor), requires_grad = True)
        else:        
            loss = Variable(torch.zeros(num_example,1).type(torch.FloatTensor), requires_grad = True)
        for i in range(1,classes):
            loss = loss +  torch.mul(target[:,i],torch.log(input[:,i])) + \
            torch.mul(1 - target[:,i],torch.log(1 - input[:,i]))

        # consider the first label 
        loss = target[:,0] * loss + (1 - target[:,0]) * torch.log(1 - input[:,0]) \
                + target[:,0] * torch.log(input[:,0])
        loss = - torch.sum(loss)
        loss = loss / num_example
        loss = loss / classes 

        return loss

class MultiLabelLoss():
    """Creates a criterion that optimizes a multi-label one-versus-all
    loss based on max-entropy, between input `x` and target `y` of size `(N, C)`.
    For each sample in the minibatch::

       loss(x, y) = - sum_i (y[i] * log( 1 / (1 + exp(-x[i])) )
                         + ( (1-y[i]) * log(exp(-x[i]) / (1 + exp(-x[i])) ) )

    where `i == 0` to `x.nElement()-1`, `y[i]  in {0,1}`.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
           class. If given, it has to be a Tensor of size `C`. Otherwise, it is
           treated as if having all ones.
        size_average (bool, optional): By default, the losses are averaged over
            observations for each minibatch. However, if the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch.
            Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on :attr:`size_average`. When
            :attr:`reduce` is ``False``, returns a loss per batch element instead and
            ignores :attr:`size_average`. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` where `N` is the batch size and `C` is the number of classes.
        - Target: :math:`(N, C)`, same shape as the input.
        - Output: scalar. If `reduce` is False, then `(N)`.
    """

    def compute(self, input, target):
        return F.binary_cross_entropy(input, target, weight=None, size_average=True)
        #multilabel_soft_margin_loss(input, target, self.weight, self.size_average,self.reduce)
