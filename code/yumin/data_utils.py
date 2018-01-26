"""Data utility functions."""
import os,sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
from tqdm import tqdm
import scipy.misc
import torchvision
import torchvision.transforms as transforms
from shutil import copyfile
import glob

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

def norm_split_data(data,num_classes):
    """
       Normalize data with mean and std, then split in train,val,test.
       aug_data: augmented Cancer dataset, data.X is a list of torch Tensor in scale range [0,1], data.y is a list.
       Return a tuple of train-, val-, test- dataset, and train_aug_fractions.
    """
    # Normalize dataset
    print('Nomralize data ...')
    mean = np.mean(data.X)
    print('mean:{}'.format(mean))
    normData = data.X - mean
    std = np.std(data.X)
    print('std:{}'.format(std))
    normData = normData/std
    print('Done nomralize data.')

    print('Splitting dataset...')   
    
    # Compute dataset class fractions
    class_statistics = []
    class_fractions =[]
    for i in range(num_classes):
        class_statistics.append(0)
        class_fractions.append(0)
        
    for label in data.y:
        class_statistics[label] = class_statistics[label]+1
        
    for i in range(num_classes):
        class_fractions[i] = class_statistics[i]/len(data)
        
    print('class_fractions:',class_fractions)

    # Split the data set into Train,Val,Test with 0.7,0.1,0.2
    y = data.y

    num_training = total*0.7
    num_training = int(np.ceil(num_training))

    num_validation=total*0.1
    num_validation = int(np.ceil(num_validation))

    mask = range(num_training)
    X_train = normData[mask]
    y_train = y[mask]
    mask = range(num_training, num_training + num_validation)
    X_val = normData[mask]
    y_val = y[mask]
    mask = range(num_training + num_validation,len(data))
    X_test = normData[mask]
    y_test = y[mask]
    print('OK...')

    return (CancerData(X_train, y_train),
            CancerData(X_val, y_val),
            CancerData(X_test, y_test),
            train_aug_fractions
            )


def data_augmentation1(data,img_name_list,num_classes):
    for i in range(num_classes):
        augmentorFiles=glob.glob('/home/ubuntu/dl4cvproject/data/train_classes/'+i+'/output/*.JPEG')
        img_name_list.append(img_name)
        img = scipy.misc.imread(fullname)
    
    
    return data

def data_augmentation2(data,img_name_list,fractions):

    """
       Augment image data with probability by defining step in loop: for i in range(start,stop,step)
       data: Cancer dataset, data.X is a list of torch Tensor in scale range [0,1], data.y is a list.
       Return augmented dataset without normalization.
    """

    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    original_length = len(data)

    transform = transforms.Compose([transforms.ToPILImage()])
    trsfmToTensor = transforms.ToTensor()
    transform10 =transforms.Compose([transforms.ToPILImage(),
                                     transforms.ColorJitter(brightness=0.5),
                                     transforms.RandomCrop(224), 
                                     transforms.Resize(256),
                                     ])

    #for i in range(original_length):
    for i in range(10):
        print(i)
        sample,label = data[i]
        print(type(sample),sample.shape)
        num_aug = int(np.floor((6000-1000-statistics[label]*2)/(statistics[label]*2)))
        print('augment image i = ',i)
        for j in range(num_aug-1):
            aug_PIL = transform10(sample)
            aug_torchTensor = trsfmToTensor(aug_PIL)
            print(type(aug_torchTensor))
            print(aug_torchTensor.shape)
            data.X.append(aug_torchTensor)
            data.y.append(label)
        print("=============================================")


    if len(data.X) != len(data.y):
        print('Error in data augmentation, dimension of X and y not same')
    else:
        print('OK...')
        
    return data

# 2.Step
def devide_dataset_in_class_folders_and_duplicate_small_classes(data,img_name_list,num_classes,fractions):
    """
    Copy the images in folder train256 to corresponding class folder in '/home/ubuntu/dl4cvproject/data/train_classes/'.
    Duplicate image i/data.X[i]/data.y[i], if i is from small class.
    
    Return a Cancer dataset, which appended with duplicated data.
    """
    for i in range(num_classes):
        print('Make train_classes dir:')
        if not os.path.exists('/home/ubuntu/dl4cvproject/data/train_classes/'+str(i)):
            os.makedirs('/home/ubuntu/dl4cvproject/data/train_classes/'+str(i))
    print('Make dir prepare finished...')
    
    original_length = len(data)
    
    for i in range(original_length):
        # Move
        sample,label=data[i]
        print('label: ',label)
        src = os.path.join('/home/ubuntu/dl4cvproject/data/train256',img_name_list[i])
        dst1 = '/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/'+ img_name_list[i]
        print('Copy from ',src,' to ',dst1,' ...')
        copyfile(src,dst1)
        data.X.append(sample)
        data.y.append(label)
        # Duplicate
        dst2='/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/2nd'+ img_name_list[i]
        copyfile(src,dst2)
        print('Copy from ',src,' to ',dst2,' ...')
        dup_img_name = '2nd'+ img_name_list[i]
        img_name_list.append(dup_img_name)
        data.X.append(sample)
        data.y.append(label)
        """
        print('duplicate small classes...')
        if fractions[label]<0.02 :
            print('duplicate small classes 5 times:',label)
            dst2='/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/2nd'+ img_name_list[i]
            dst3='/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/3rd'+ img_name_list[i]
            dst4='/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/4th'+ img_name_list[i]
            dst5='/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/5th'+ img_name_list[i]
            dst6='/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/6th'+ img_name_list[i]
            copyfile(src,dst2)
            copyfile(src,dst3)
            copyfile(src,dst4)
            copyfile(src,dst5)
            copyfile(src,dst6)
            print('Copy from ',src,' to ',dst2,' ...')
            print('Copy from ',src,' to ',dst3,' ...')
            print('Copy from ',src,' to ',dst4,' ...')
            print('Copy from ',src,' to ',dst5,' ...')
            print('Copy from ',src,' to ',dst6,' ...')
            for c in range(4):
                print('append:',c)
                data.X.append(sample)
                data.y.append(label)
            
        if fractions[label]<0.04 and fractions[label]>0.02 :
            print('duplicate small classes 4 times:',label)
            dst2='/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/2nd'+ img_name_list[i]
            dst3='/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/3rd'+ img_name_list[i]
            dst4='/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/4th'+ img_name_list[i]
            dst5='/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/5th'+ img_name_list[i]
            copyfile(src,dst2)
            copyfile(src,dst3)
            copyfile(src,dst4)
            copyfile(src,dst5)
            print('Copy from ',src,' to ',dst2,' ...')
            print('Copy from ',src,' to ',dst3,' ...')
            print('Copy from ',src,' to ',dst4,' ...')
            print('Copy from ',src,' to ',dst5,' ...')
            for c in range(3):
                print('append:',c)
                data.X.append(sample)
                data.y.append(label)
                
        if fractions[label]<0.06 and fractions[label]>0.04 :
            print('duplicate small classes 3 times:',label)
            dst2='/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/2nd'+ img_name_list[i]
            dst3='/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/3rd'+ img_name_list[i]
            dst4='/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/4th'+ img_name_list[i]

            copyfile(src,dst2)
            copyfile(src,dst3)
            copyfile(src,dst4)

            print('Copy from ',src,' to ',dst2,' ...')
            print('Copy from ',src,' to ',dst3,' ...')
            print('Copy from ',src,' to ',dst4,' ...')

            for c in range(2):
                print('append:',c)
                data.X.append(sample)
                data.y.append(label)

        if fractions[label]<0.08and fractions[label]>0.06:
            print('duplicate small classes twice:',label)
            dst2='/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/2nd'+ img_name_list[i]
            dst3='/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/3rd'+ img_name_list[i]

            copyfile(src,dst2)
            copyfile(src,dst3)

            print('Copy from ',src,' to ',dst2,' ...')
            print('Copy from ',src,' to ',dst3,' ...')

            for c in range(1):
                print('append:',c)
                data.X.append(sample)
                data.y.append(label)
                
        if fractions[label]<0.1 and fractions[label]>0.08:
            print('duplicate small classes 1 time:',label)
            dst2='/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/2nd'+ img_name_list[i]

            copyfile(src,dst2)

            print('Copy from ',src,' to ',dst2,' ...')

            print('append:')
            data.X.append(sample)
            data.y.append(label)
        """                      
    print('OK...')
    return duplicate_data

# 1.Step
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
    img_name_list = []
    num_bad = 0
    good_mask = []
    idx = 0

    for img_name in tqdm(csv['image_name'].values):
        fullname = os.path.join(img_folder_full_name, img_name)
        img_name_list.append(img_name)
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
        #if idx > 100:
        #    break
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
        #if idx > 100:
        #    break
        idx = idx + 1

    for i in range(14):
        class_fractions[i] = class_statistics[i]/total_GoodImg

    print('OK...')

    return CancerData(img_list, label_list),img_name_list,class_statistics,class_fractions
