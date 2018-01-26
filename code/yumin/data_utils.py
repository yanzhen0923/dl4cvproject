"""Data utility functions."""
import os
import matplotlib.pyplot as plt
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

        #img = torch.from_numpy(img).float()
        return img, label

    def __len__(self):
        return len(self.y)

def norm_split_data(aug_data):
    print('Nomralize data ...')
    mean = np.mean(aug_data.X)
    print('mean:{}'.format(mean))
    normData = aug_data.X - mean
    std = np.std(aug_data.X)
    print('std:{}'.format(std))
    normData = normData/std
    print('Done nomralize data')

    print('Splitting dataset...')

    # Split the data set into Train,Val,Test with 0.8,0.1,0.1
    y = aug_data.y
    total = len(y)

    num_training = total*0.8
    num_training = int(np.ceil(num_training))

    num_validation=total*0.1
    num_validation = int(np.ceil(num_validation))

    mask = range(num_training)
    X_train = normData[mask]
    y_train = y[mask]
    mask = range(num_training, num_training + num_validation)
    X_val = normData[mask]
    y_val = y[mask]
    mask = range(num_training + num_validation,total)
    X_test = normData[mask]
    y_test = y[mask]
    print('OK...')

    return (CancerData(torch.from_numpy(X_train).float(), y_train),
            CancerData(torch.from_numpy(X_val).float(), y_val),
            CancerData(torch.from_numpy(X_test).float(), y_test),
            )


def data_augmentation(data,fractions):

    """
       Augment image data.
       data.X = a list of torch Tensor in scale range [0,1]
       Return aug. dataset incl. augmented data but without normalization.
    """

    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    original_length = len(data)

    transform = transforms.Compose([transforms.ToPILImage()])
    trsfmToTensor = transforms.ToTensor()
    transform10 =transforms.Compose([transforms.ToPILImage(), 
                                     #transforms.ColorJitter(brightness = 0.1,contrast = 0.1),
                                     transforms.RandomCrop(224), 
                                     transforms.Resize(256),
                                     ])
    
    for i in range(original_length):
    #for i in range(10):
        print(i)
        sample,label = data[i]
        print(type(sample),sample.shape)
        if fractions[label] < 0.05:
            print('augment image i = ',i)
            #for j in range(10):
            aug_PIL = transform10(sample)
            aug_torchTensor = trsfmToTensor(aug_PIL)
        
            print(type(aug_torchTensor))
            print(aug_torchTensor.shape)
            data.X.append(aug_torchTensor)
            data.y.append(label)

            # To show original image, convert to PIL image first
            #img = transform(sample)
            #plt.imshow(img)
            #plt.show()
            print("=============================================")
        else:
            print('OK')

    if len(data.X) != len(data.y):
        print('Error in data augmentation, dimension of X and y not same')
    else:
        print('OK...')
    return data

def duplicate_small_class(data,class_statistics):
    """
    Duplicate small class, so that all class have same size, i.e. balanced dataset. No normalization.

    data.X: a list of torch tensor, scale range [0,1]
    data.y: a list
    class_statistics: orig dataset statistics
    """
    
    
    
    
    return duplicate_data


# train, val, test partition in folder train256
# In total, 18577 images in train256
def read_cancer_dataset(csv_full_name,
                        img_folder_full_name):

    """
    Load and preprocess the Cancer dataset, throw out bad data.
    
    Return class statistics and dataset without augmentation nor normalization.
    img_list: a list of torch tensor, scale range [0,1]
    label_list: a list
    class_statistics: number of samples in each class
    class_fractions: fraction of each class in whole dataset
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
            # repeat channels
            one_channel_img = np.expand_dims(img, axis=2)
            # numpy H x W x C
            three_channel_img = np.repeat(one_channel_img, 3, axis=2)
            # convert numpy to torch tensor, and scale range to [0,1]
            trsfmToTensor = transforms.ToTensor()
            three_channel_img_tensor = trsfmToTensor(three_channel_img)
            img_list.append(three_channel_img_tensor)
            good_mask.append(True)

        else:
            good_mask.append(False)
            num_bad = num_bad + 1
            print('bad image: ',fullname,'total bad images: ',num_bad)
        # This if only for debug
        if idx > 100:
            break
        idx =  idx + 1

    total_GoodImg = idx + 1 - num_bad
    print('Total good data size: ',total_GoodImg)

    class_statistics = []
    class_fractions =[]
    for i in range(14):
        class_statistics.append(0)
        class_fractions.append(0)

    label_list = []
    idx = 0
    for class_str in tqdm(csv['detected'].values):
        if good_mask[idx] == True:
            label = int(class_str[6:]) - 1
            class_statistics[label] = class_statistics[label]+1
            label_list.append(label)
        if idx > 100:
            break
        idx = idx + 1

    for i in range(14):
        class_fractions[i] = class_statistics[i]/total_GoodImg

    print('OK...')

    return CancerData(img_list, label_list),class_statistics,class_fractions
