"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
from tqdm import tqdm
import scipy.misc
from torchvision.transforms import *

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
        # all data pre-processing are done before
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)
    
def get_balanced_weights(label_list, num_classes):
    # count class appearance
    count = [0] * num_classes                                                      
    for label in label_list:                                                         
        count[label] += 1
    
    # compute class weight
    weight_per_class = [0.] * num_classes                                      
    N = float(sum(count))                                                   
    for i in range(num_classes):                                                   
        weight_per_class[i] = N/float(count[i])
    
    #assign weights for each data entry
    weights = [0] * len(label_list)                                     
    for idx, label in enumerate(label_list):                                          
        weights[idx] = weight_per_class[label]
        
    return weights          
    


def crop_center(img, cropx, cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]
    
def get_Cancer_datasets(csv_full_name, img_folder_full_name, num_training=16000, num_validation=1250,
                         num_test=1327, mode='train', dtype=np.float32):
    
    debug_cnt = 0
    """
    Load and preprocess the Cancer dataset.
    """
    csv = pd.read_csv(csv_full_name)
    
    img_list = []
    original_img_list = []
    for img_name in tqdm(csv['image_name'].values):
        debug_cnt += 1
        fullname = os.path.join(img_folder_full_name, img_name)
        img = scipy.misc.imread(fullname)
        if len(img.shape) != 2:
            # 4 channels to 1 channel
            img = np.squeeze(np.delete(img, (1, 2, 3), 2))
        
        #crop the one channel image
        img = crop_center(img, 280, 280)
        
        #random flip
        coin = np.random.randint(0, 10)
        if coin > 4:
            img = np.fliplr(img)
        
        # record to compute mean and std
        original_img_list.append(img)
        
        # repeat channels
        one_channel_img = np.expand_dims(img, axis=0)
        three_channel_img = np.repeat(one_channel_img, 3, axis=0)
        
        #add to final list
        img_list.append(three_channel_img)
        
        if debug_cnt == num_training + num_validation + num_test:
            break
        
    print('transforming...')
    X = np.array(img_list)
    X_original = np.array(original_img_list)
    print('X.shape:{}'.format(X.shape))
    print('X_original.shape:{}'.format(X_original.shape))
    
    # Normalize the data
    mean = np.mean(X_original)
    print('mean:{}'.format(mean))
    std = np.std(X_original)
    print('std:{}'.format(std))

    X = X.astype(float)
    X -= mean
    X /= std
    print('Done transforming...')
    
    # return x and ids in csv
    if mode != 'train':
        return torch.autograd.Variable(torch.from_numpy(X).float(), requires_grad=False), csv
        
    print('Getting labels')
    label_list = []
    for class_str in tqdm(csv['detected'].values):
        label_list.append(int(class_str[6:]) - 1)
    y = np.array(label_list)
    
    print('submasking...')
    
    # Subsample the data
    print('num_training:{}'.format(num_training))
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
    
    return (CancerData(torch.from_numpy(X_train).float(), y_train),
            CancerData(torch.from_numpy(X_val).float(), y_val),
            CancerData(torch.from_numpy(X_test).float(), y_test),
            y_train)
