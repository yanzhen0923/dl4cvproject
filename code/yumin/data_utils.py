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

def norm_split_data(data,fractions):
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
            fractions
            )

#4th.Step, before this run ImgAugmentation.py in terminal.
def append_augmented_data(data,statistics):
    length=len(data)
    old_total=sum(statistics)
    if length != old_total:
        print('Error in append_augmented_data 1...')
        return
    
    num_classes=len(statistics)
    new_fractions = [0.0]*num_classes
    
    for c in range(num_classes):
        files=glob.glob('/home/ubuntu/dl4cvproject/data/train_classes/'+c+'/output/*.JPEG')
        for f in files:
            img = scipy.misc.imread(f)
            data.X.append(img)
            data.y.append(c)
            statistics[c]=statistics[c]+1
            print(f,' is appended to dataset with label ',c)
    new_total=sum(statistics)

    for i in range(num_classes):
        new_fractions[i]=statistics[i]/new_total
    
    if new_total != len(data):
        print('Error in append_augmented_data 2...')
    else:
        print(new_total-old_total,' new data appended to dataset...')
        print('New statistics after run ImgAugmentor.py:',statistics)
        print('New fractions after run ImgAugmentor.py:',new_fractions)
        print('append_augmented_data is OK...')
        
    return data,statistics,new_fractions

# 5th. Step
def data_augmentation(data,statistics):

    """
       Augment image data with probability by defining step in loop: for i in range(start,stop,step)
       data: Cancer dataset, data.X is a list of torch Tensor in scale range [0,1], data.y is a list.
       Return augmented dataset without normalization.
    """
    
    """
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    """
    length=len(data)
    old_total=sum(statistics)
    if length != old_total:
        print('Error in data_augmentation 1...')
        return
    else:
        print('Start brightness,randomcrop augmentation...')
    
    trsfmToPIL= transforms.Compose([transforms.ToPILImage()])
    trsfmToTensor = transforms.ToTensor()
    transform =transforms.Compose([transforms.ToPILImage(),
                                     transforms.ColorJitter(brightness=0.5),
                                     transforms.RandomCrop(224), 
                                     transforms.Resize(256),
                                     ])
    max_class_size = max(statistics)
    num_classes = len(statistics)
    num_aug = [0]*num_classes
    num_trsfm_for_each_img =[0]*num_classes
    new_fractions =[0.0]*num_classes
    
    for c in range(num_classes):
        num_aug[c] = max_class_size-statistics[c]
        num_trsfm_for_each_img[c] = int(np.floor(num_aug[c]/statistics[c]))-1
        print('For class ',c,', we need ',num_trsfm_for_each_img[c],' new augmented images')
    
    for i in range(old_total):
    #for i in range(10):
        print(i)
        sample,label = data[i]
        print(type(sample),sample.shape)
        
        print('augment image i = ',i)
        for j in range(num_trsfm_for_each_img[c]):
            aug_PIL = transform(sample)
            aug_torchTensor = trsfmToTensor(aug_PIL)
            print(type(aug_torchTensor))
            print(aug_torchTensor.shape)
            data.X.append(aug_torchTensor)
            data.y.append(label)
            statistics = statistics + 1
        print("=============================================")
    new_total = sum(statistics)
    for i in range(num_classes):
        new_fractions = statistics/new_total
        
    if len(data.X) != len(data.y):
        print('Error in data augmentation 2...')
    elif len(data) != new_total:
        print('Error in data augmentation 3...')
    else:
        print(new_total-old_total,' new data appended to dataset...')
        print('New statistics after brightness,randomcrop augmentation:',statistics)
        print('New fractions  after brightness,randomcrop augmentation:',new_fractions)
        print('Brightness,randomcrop augmentation is OK...')   
        return data, statistics,new_fractions

# 2.Step
def devide_dataset_in_class_folders_and_duplicate_small_classes(data,old_fractions,statistics):
    """
    Copy the images in folder train256 to corresponding class folder in '/home/ubuntu/dl4cvproject/data/train_classes/'.
    Duplicate image i/data.X[i]/data.y[i], if i is from small class.
    
    Return a Cancer dataset, which appended with duplicated data.
    """
    length=len(data)
    old_total=sum(statistics)
    if length != old_total:
        print('Error in devide_dataset_in_class_folders_and_duplicate_small_classes 1...')
        return
    else:
        print('Start devide_dataset_in_class_folders_and_duplicate_small_classes...')
    
    num_classes = len(statistics)
    
    for i in range(num_classes):
        print('Make train_classes dir:')
        if not os.path.exists('/home/ubuntu/dl4cvproject/data/train_classes/'+str(i)):
            os.makedirs('/home/ubuntu/dl4cvproject/data/train_classes/'+str(i))
    print('Make dir finished...')

    new_fractions =[0.0]*num_classes
    
    for i in range(old_total):
        # Move i
        sample,label=data[i]
        print('label: ',label)
        src = os.path.join('/home/ubuntu/dl4cvproject/data/train256',img_name_list[i])
        dst1 = '/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/'+ img_name_list[i]
        print('Move from ',src,' to ',dst1,' ...')
        copyfile(src,dst1)

        # Duplicate i
        if fractions[label]<0.1:
            dst2='/home/ubuntu/dl4cvproject/data/train_classes/'+str(label)+'/2nd'+ img_name_list[i]
            copyfile(src,dst)
            print('Duplicate from ',src,' to ',dst2,' ...')
            data.X.append(sample)
            data.y.append(label)
            statistics[label]=statistics[label]+1
            
    print('End for loop...')
              
    new_total= sum(statistics)

    for i in range(num_classes):
        new_fractions[i] = statistics[i]/new_total
        
    if len(data.X) != len(data.y):
        print('Error in devide_dataset_in_class_folders_and_duplicate_small_classes 2...')
        return
    elif len(data) != new_total:
        print('Error in devide_dataset_in_class_folders_and_duplicate_small_classes 3...')
        return
    else:
        print(new_total-old_total,' new data appended to dataset...')
        print('statistics after duplicate:',statistics)
        print('fractions after duplicate:',new_fractions)
        print('Devide_dataset_in_class_folders_and_duplicate_small_classes is OK...')
        return duplicate_data,statistics,new_fractions

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
    img_name_list=[]
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
            img_name_list.append(img_name)

        else:
            good_mask.append(False)
            num_bad = num_bad + 1
            os.remove(fullname)
            print('delete bad image ',fullname)
        # This if only for debug
        #if idx > 100:
        #    break
        idx =  idx + 1

    total_GoodImg = idx + 1 - num_bad
    print('Total good data size: ',total_GoodImg)

    class_statistics = [0]*num_classes
    class_fractions =[0.0]*_num_classes

    label_list = []
    idx = 0
    for class_str in tqdm(csv['detected'].values):
        if good_mask[idx] == True:
            label = int(class_str[6:]) - 1
            class_statistics[label] = class_statistics[label]+1
            label_list.append(label)
        else:
            print('A bad image does not append label to label_list...')
        
        #if idx > 100:
        #    break
        idx = idx + 1

    for i in range(14):
        class_fractions[i] = class_statistics[i]/total_GoodImg
        
    if len(data.X) != len(data.y):
        print('Error in read_cancer_dataset 1...')
        return
    elif total_GoodImg != sum(class_statistics):
        print('Error in read_cancer_dataset 2...')
        return
    elif len(data) != total_GoodImg:
        print('Error in read_cancer_dataset 3...')
        return
    else:
        print('statistics after read_cancer_dataset:',class_statistics)
        print('fractions after read_cancer_dataset:',class_fractions)
        print('Read_cancer_dataset is OK...')
        return CancerData(img_list, label_list),class_statistics,class_fractions,img_name_list
