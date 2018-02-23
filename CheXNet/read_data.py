# encoding: utf-8

"""
Read images and corresponding labels.
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# borrowed from http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
    transforms.Resize(224),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    transforms.ToTensor()])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize(224),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor

#TRAIN_DATA_DIR = 'images/train'
#TRAIN_IMAGE_LIST = 'train_list.txt'

#DEV_DATA_DIR = 'images/dev' 
#DEV_IMAGE_LIST = 'dev_list.txt'

TRAIN_BATCH_SIZE = 11

class ChestXrayDataSet(Dataset):
    def __init__(self, image_file, label_file, transform=None):
        """
        Args:
            image_file: path to image files.
            label_file: output path to the file containing corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(label_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]                
                # we use binary classification, thus we only choose item[1]
                #label = items[1]
                finding = np.count_nonzero( np.asarray(items[1:], dtype = np.int32)  )
                if finding == 0:
                    label = [0]
                else:
                    label = [1]
                # label = items[1:]
                #label = [int(i) for i in label]
                image_name = os.path.join(image_file, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.Tensor(label)

    def __len__(self):
        return len(self.image_names)


def fetch_dataloader(types, image_dir, label_dir):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        image_dir: (string) directory containing all the images
        label_dir: (string) directory containing all the labels
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'dev', 'test']:
        if split in types:
            label_path = os.path.join(label_dir, "{}_list.txt".format(split))
            image_path = os.path.join(image_dir, "{}".format(split))
            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                ds = ChestXrayDataSet(image_file=image_path, label_file=label_path, transform=train_transformer)
            else:
                ds = ChestXrayDataSet(image_file=image_path, label_file=label_path, transform=eval_transformer)
            dataloaders[split] = DataLoader(ds, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=False)

    return dataloaders

