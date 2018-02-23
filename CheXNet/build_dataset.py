import argparse
import random
import os
import torch

import pandas as pd
from skimage import io, transform
import numpy as np

from PIL import Image
from tqdm import tqdm

SIZE = 224
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

parser = argparse.ArgumentParser()
if (not torch.cuda.is_available()):
   parser.add_argument('--data_dir', default='ChestX-ray14/images', help="Directory with the X-ray image dataset")
   parser.add_argument('--output_image_dir', default='images/', help="Where to write the new images")
   parser.add_argument('--output_label_dir', default='labels/', help="Where to write the new labels")
else:
   parser.add_argument('--data_dir', default='/home/ubuntu/Dataset/images5', help="Directory with the X-ray image dataset")
   parser.add_argument('--output_image_dir', default='/home/ubuntu/Data_Processed/images/', help="Where to write the new images")
   parser.add_argument('--output_label_dir', default='/home/ubuntu/Data_Processed/labels/', help="Where to write the new labels")

def split_images(filenames, perc1, perc2, perc3):
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split1 = int(perc1 * len(filenames))
    train_filenames = filenames[:split1]
    filenames = filenames[split1:]
    split2 = int(perc2 / (perc2 + perc3) * len(filenames))
    dev_filenames = filenames[:split2]
    test_filenames = filenames[split2:]

    return train_filenames, dev_filenames, test_filenames

def process_CSV_file(filename):
    df = pd.read_csv(filename)  # read in the csv label file
    # split the labels
    labels = df['Finding Labels'].str.split('|',expand=True)
    # insert index column at first
    labels['Image Index'] = df['Image Index']
    cols = labels.columns.tolist()
    df = labels[[cols[-1]] + cols[:-1]]
    return df 

def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))

def write_file_to_list(filename, listfile, datafile):
    """datafile: object of dataframe"""
    # get the row corresponding to the filename
    filename = filename.split('/')[-1]
    row = datafile.loc[filename]
    
    # get all labels
    labels = np.zeros((len(CLASS_NAMES)))
    for j in range(len(CLASS_NAMES)):
        for k in range(len(row)):
            if row[k] == CLASS_NAMES[j]:
                labels[j] = 1
    # wrtie in
    listfile.write("%s " % filename)
    for label in labels:
        listfile.write("%s " % int(label))
    listfile.write('\n')

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(args.data_dir)
    filenames = [os.path.join(args.data_dir, f) for f in filenames if f.endswith('.png')]

    # Split the images in 'train_signs' into 80% train, 10% dev and 10% test
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    train_filenames, dev_filenames, test_filenames = split_images(filenames, 0.7, 0.2, 0.1)

    filenames = {'train': train_filenames,
                 'dev': dev_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_image_dir):
        os.mkdir(args.output_image_dir)
    else:
        print("Warning: output image dir {} already exists".format(args.output_image_dir))

    if not os.path.exists(args.output_label_dir):
        os.mkdir(args.output_label_dir)
    else:
        print("Warning: output labeldir {} already exists".format(args.output_label_dir))

    # Preprocess train, val and test
    datafile = process_CSV_file("Data_Entry_2017.csv")
    datafile = datafile.set_index(["Image Index"]) # use the column named 'Image Index'
    for split in ['train', 'dev', 'test']:
        # get the directory of output images
        output_image_dir_split = os.path.join(args.output_image_dir, '{}'.format(split))
        if not os.path.exists(output_image_dir_split):
            os.mkdir(output_image_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_image_dir_split))

        print("Processing {} data, saving preprocessed images to {} and writing corresponding list to {}".
                format(split, output_image_dir_split, args.output_label_dir))
        # open the list file
        listname = os.path.join(args.output_label_dir, '{}'.format(split + "_list.txt"))
        listfile = open(listname, 'w')
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_image_dir_split, size=SIZE)
            write_file_to_list(filename, listfile, datafile)
        listfile.close()

    print("Done building dataset")
