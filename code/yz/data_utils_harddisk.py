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

def get_3_channel_image(img):
    if len(img.shape) != 2:
        img = np.squeeze(np.delete(img, (1, 2, 3), 2))
        
    one_channel_img = np.expand_dims(img, axis=2)
    three_channel_img = np.repeat(one_channel_img, 3, axis=2)
    
    return three_channel_img

class CancerDataTrain(data.Dataset):

    def __init__(self, root, img_name_list, label_list):
        self.label_list = label_list
        self.img_name_list = img_name_list
        self.root = root
        self.transform = Compose([
                            ToPILImage(),
                            RandomCrop(224),
                            RandomHorizontalFlip(),
                            ColorJitter(brightness=0.3, contrast=0.3),
                            ToTensor()
                            ])
        
    def __getitem__(self, index):
        
        fullname = os.path.join(self.root, self.img_name_list[index])
        img = scipy.misc.imread(fullname)
        img = get_3_channel_image(img)
        
        return self.transform(img), self.label_list[index]

    def __len__(self):
        return len(self.label_list)

class CancerDataVT(data.Dataset):

    def __init__(self, root, img_name_list, label_list):
        self.label_list = label_list
        self.img_name_list = img_name_list
        self.root = root
        self.transform = Compose([
                            ToPILImage(),
                            CenterCrop(224),
                            ToTensor()
                            ])
        
    def __getitem__(self, index):
        
        fullname = os.path.join(self.root, self.img_name_list[index])
        img = scipy.misc.imread(fullname)
        img = get_3_channel_image(img)
        
        return self.transform(img), self.label_list[index]

    def __len__(self):
        return len(self.label_list)

    
class CancerDataUpload(data.Dataset):

    def __init__(self, root, img_name_list):
        self.img_name_list = img_name_list
        self.root = root
        self.transform = Compose([
                            ToPILImage(),
                            CenterCrop(224),
                            ToTensor()
                            ])
        
    def __getitem__(self, index):
        
        fullname = os.path.join(self.root, self.img_name_list[index])
        img = scipy.misc.imread(fullname)
        img = get_3_channel_image(img)
        
        return self.transform(img)

    def __len__(self):
        return len(self.img_name_list)

    
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
    img_name_list = csv['image_name'].tolist()
    
    if mode != 'train':
        return CancerDataUpload(img_folder_full_name, img_name_list), csv
    
    label_list = []
    for class_str in tqdm(csv['detected'].values):
        label_list.append(int(class_str[6:]) - 1)
    
    print('num_training:{}'.format(num_training))
    print('num_validation:{}'.format(num_validation))
    print('num_test:{}'.format(num_test))
    
    img_name_train = img_name_list[0:num_training]
    label_train = label_list[0:num_training]
    img_name_val = img_name_list[num_training:num_training + num_validation]
    label_val = label_list[num_training:num_training + num_validation]
    img_name_test = img_name_list[num_training + num_validation:-1]
    label_test = label_list[num_training + num_validation:-1]
    
    print('OK...')
    
    return (CancerDataTrain(img_folder_full_name, img_name_train, label_train),
            CancerDataVT(img_folder_full_name, img_name_val, label_val),
            CancerDataVT(img_folder_full_name, img_name_test, label_test),
            label_train)
