"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
from tqdm import tqdm
import scipy.misc

import torchvision
import torchvision.transforms as transforms

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

#def data_augmentation(data,labels,transform):





# train, val, test partition in folder train256
# In total, 18577 images in train256
def get_Cancer_datasets(csv_full_name,
                        img_folder_full_name):
    
    """
    Load and preprocess the Cancer dataset.
    """
    csv = pd.read_csv(csv_full_name)
    
    img_list = []

    num_bad = 0
    good_mask = []
    idx = 0

    for img_name in tqdm(csv['image_name'].values):
        
        fullname = os.path.join(img_folder_full_name, img_name)
        img = scipy.misc.imread(fullname)

        if len(img.shape) == 2:
            img_list.append(np.expand_dims(img, axis=0))
            good_mask.append(True)

        else:
            good_mask.append(False)
            num_bad = num_bad + 1
            print('bad image: ',fullname,'total bad images: ',num_bad)
        # This if only for debug
        if idx > 1000:
            break
        idx =  idx + 1

    total_GoodImg = idx + 1 - num_bad
    print('Total good data size: ',total_GoodImg)
    
    print('Scale to [0,1]...')
    X = np.array(img_list)
    print(X.shape,type(X))
    X = X / 255.0
    print('Done Scale to [0,1]')
    #each_mean = np.mean(X, axis=(2,3))
    #mean = np.mean(each_mean,axis = 0)
    #std = np.std(X,axis=(0, 2, 3))

    class_statistics = []
    for i in range(14):
        class_statistics.append(0)
        print(class_statistics[i])

    label_list = []
    idx = 0
    for class_str in tqdm(csv['detected'].values):
        if good_mask[idx] == True:
            label = int(class_str[6:]) - 1
            class_statistics[label] = class_statistics[label]+1
            label_list.append(label)
        if idx > 1000:
            break
        idx = idx + 1

    y = np.array(label_list)
    print('label shape',y.shape)
    #y = y[good_mask]
    
    # print('submasking...')
    
    # Subsample the data
    #
    # num_training=total_GoodImg*0.8
    # num_training = int(np.ceil(num_training))
    #
    # num_validation=total_GoodImg*0.1
    # num_validation = int(np.ceil(num_validation))
    #
    # num_test= total_GoodImg - num_training-num_validation
    #
    # mask = range(num_training)
    # X_train = X[mask]
    # y_train = y[mask]
    # mask = range(num_training, num_training + num_validation)
    # X_val = X[mask]
    # y_val = y[mask]
    # mask = range(num_training + num_validation,
    #              num_training + num_validation + num_test)
    # X_test = X[mask]
    # y_test = y[mask]
    
    print('OK...')

    return CancerData(X, y)
         # (CancerData(X_train, y_train),
            # CancerData(X_val, y_val),
            # CancerData(X_test, y_test),)
