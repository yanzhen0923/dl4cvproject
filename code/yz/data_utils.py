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

    
class CancerDataTrain(data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.transform = Compose([
                            ToPILImage(),
                            RandomCrop(224),
                            RandomHorizontalFlip(),
                            ColorJitter(brightness=0.3, contrast=0.3),
                            ToTensor()
                            ])
        
    def __getitem__(self, index):
        return self.transform(self.X[index]), self.y[index]

    def __len__(self):
        return len(self.y)

class CancerDataVT(data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.transform = Compose([
                            ToPILImage(),
                            CenterCrop(224),
                            ToTensor()
                            ])
        
    def __getitem__(self, index):
        return self.transform(self.X[index]), self.y[index]

    def __len__(self):
        return len(self.y)
    
class CancerDataUpload(data.Dataset):

    def __init__(self, X):
        self.X = X
        self.transform = Compose([
                            ToPILImage(),
                            CenterCrop(224),
                            ToTensor()
                            ])
        
    def __getitem__(self, index):
        return self.transform(self.X[index])

    def __len__(self):
        return self.X.shape[0]
    
def get_balanced_weights(label_list, num_classes, factor=0.7):
    # count class appearance
    count = [0] * num_classes                                                      
    for label in label_list:                                                         
        count[label] += 1
    
    # compute class weight
    weight_per_class = [0.] * num_classes                                      
    N = float(sum(count))                                                   
    for i in range(num_classes):                                                   
        weight_per_class[i] = 100 / (72 * np.power(float(count[i]), factor))
        
    print('weights: {}'.format(weight_per_class))
    print('equivalent_num:')
    for i in range(len(count)):
        print(weight_per_class[i] * count[i])
    
    #assign weights for each data entry
    weights = [0] * len(label_list)                                     
    for idx, label in enumerate(label_list):                                          
        weights[idx] = weight_per_class[label]
        
    return weights          
    


#def crop_center(img, cropx, cropy):
#    y,x = img.shape
#    startx = x//2-(cropx//2)
#    starty = y//2-(cropy//2)    
#    return img[starty:starty+cropy,startx:startx+cropx]
    
def get_Cancer_datasets(csv_full_name, img_folder_full_name, num_training=13000, num_validation=1857,
                         num_test=3720, mode='train', dtype=np.float32):
    
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
            
        one_channel_img = np.expand_dims(img, axis=2)
        three_channel_img = np.repeat(one_channel_img, 3, axis=2)
        
        #add to final list
        img_list.append(three_channel_img)
        
        if debug_cnt == num_training + num_validation + num_test:
            break
        
    print('transforming...')
    X = np.array(img_list)
    print('Done transforming...')
    
    if mode != 'train':
        return CancerDataUpload(X), csv
        
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
    
    return (CancerDataTrain(X_train, y_train),
            CancerDataVT(X_val, y_val),
            CancerDataVT(X_test, y_test),
            y_train)
