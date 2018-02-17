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

from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score


# CKPT_PATH = 'model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = './ChestX-ray14/images'
TRAIN_IMAGE_LIST = './ChestX-ray14/labels/try1.txt'
BATCH_SIZE = 4
use_gpu = torch.cuda.is_available()


normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

# Create the input data pipeline
logging.info("Loading the datasets...")

train_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                image_list_file=TRAIN_IMAGE_LIST,
                                transform=transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.ToTensor(),
                                ]))
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=8, pin_memory=False)



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


# a general model definition, scheduler: learning rate decay    
def train_model(model, optimizer, loss_fn, num_epochs=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    #scheduler.step()
    model.train(True)  # Set model to training mode

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0
        running_accuracy = 0.0

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

                # backward + optimize 
                loss.backward()

                # performs updates using calculated gradients
                optimizer.step()

                # cutoff by 0.5
                preds = outputs >= 0.5
                preds = preds.type(torch.FloatTensor)

                # extract data from torch Variable, move to cpu, convert to
                # numpy
                preds_np = preds.data.cpu().numpy()
                labels_np = labels.data.cpu().numpy()
                running_accuracy += np.sum(preds_np == labels_np)/float(labels_np.size)


                running_loss += loss.data[0] #* inputs.size(0)

                compare = torch.eq(preds, labels)
                compare = compare.type(torch.FloatTensor)
                running_corrects += torch.sum(compare) == N_CLASSES

                t.set_postfix(loss='{:05.3f}'.format(running_loss))
                t.update()

        running_corrects = running_corrects.float().data[0]

        # Calculate the epoch loss and epoch metrics(accuracy)
        epoch_loss = running_loss / len(train_dataset)
<<<<<<< HEAD
        epoch_acc = running_corrects / len(train_dataset)
        
=======
        #epoch_acc = running_corrects / len(train_dataset)
        epoch_acc = running_accuracy / len(train_dataset)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
>>>>>>> 6795672cf91a942841148ed11919c327f7330f45

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'train', epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model   


# initialize and load the model
model = DenseNet121(N_CLASSES)
#model = torch.nn.DataParallel(model)

criterion = nn.MultiLabelSoftMarginLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
model_conv = train_model(model, optimizer, criterion, num_epochs=5)
