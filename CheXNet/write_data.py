import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np

df = pd.read_csv("Data_Entry_2017.csv")

# split the labels
labels = df['Finding Labels'].str.split('|',expand=True)
# insert index column at first
labels['Image Index'] = df['Image Index']
cols = labels.columns.tolist()
df = labels[[cols[-1]] + cols[:-1]] 

CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

filename = 'train_list.txt'
file = open(filename, 'w')

# only wrtie the first 10 now
for i in range(10):
    row = df.iloc[i,1:].as_matrix()
    labels = np.zeros((len(CLASS_NAMES)))
    for j in range(len(CLASS_NAMES)):
        for k in range(len(row)):
            if row[k] == CLASS_NAMES[j]:
                labels[j] = 1
    #print(labels)
    file.write("%s " % df.iloc[i,0])
    for label in labels:
        file.write("%s " % int(label))
    file.write('\n')

file.close()