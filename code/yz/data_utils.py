"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
from tqdm import tqdm
import scipy.misc

class OverfitSampler(object):
    """
    Sample dataset to overfit.
    """

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples

    
class CancerData(data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]

        img = torch.from_numpy(img).float()
        return img, label

    def __len__(self):
        return len(self.y)



# still imbalanced class distribution problem exists
# still imbalanced class distribution problem exists
# still imbalanced class distribution problem exists
def get_Cancer_datasets(csv_full_name, img_folder_full_name, num_training=16000, num_validation=1250,
                         num_test=1327, mode='train', dtype=np.float32):
    
    """
    Load and preprocess the Cancer dataset.
    """
    csv = pd.read_csv(csv_full_name)
    
    img_list = []
    for img_name in tqdm(csv['image_name'].values):
        fullname = os.path.join(img_folder_full_name, img_name)
        img = scipy.misc.imread(fullname)
        if len(img.shape) != 2:
            # 4 channels to 1 channel
            img = np.squeeze(np.delete(img, (1, 2, 3), 2))
        #img_list.append(np.expand_dims(img, axis=0)
        img_list.append(np.expand_dims(img, axis=0))
        
    print('transforming...')
    X = np.array(img_list)
    X = X / 255.0
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X, axis=0)
    X -= mean_image
    print('Done transforming...')
    
    # return x and ids in csv
    if mode != 'train':
        return torch.autograd.Variable(torch.from_numpy(X).float(), requires_grad=False), csv
        
    label_list = []
    for class_str in tqdm(csv['detected'].values):
        label_list.append(int(class_str[6:]) - 1)
    y = np.array(label_list)
    
    print('submasking...')
    
    # Subsample the data
    mask = range(num_training)
    X_train = X[mask]
    y_train = y[mask]
    mask = range(num_training, num_training + num_validation)
    X_val = X[mask]
    y_val = y[mask]
    mask = range(num_training + num_validation,
                 num_training + num_validation + num_test)
    X_test = X[mask]
    y_test = y[mask]
    
    print('OK...')

    return (CancerData(X_train, y_train),
            CancerData(X_val, y_val),
            CancerData(X_test, y_test),
            mean_image)
