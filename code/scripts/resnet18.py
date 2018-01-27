
# coding: utf-8

# Image classification with CNNs
# ================
# 
# The goal of this exercise is to implement a specific CNN architecture with PyTorch and train it on the CIFAR-10 image classification dataset. We will start by introducing the dataset and then implement a `nn.Module` and a useful `Solver` class. Seperating the model from the actual training has proven itself as a sensible design decision. By the end of this exercise you should have succesfully trained your (possible) first CNN model and have a boilerplate `Solver` class which you can reuse for the next exercise and your future research projects.
# 
# For an inspiration on how to implement a model or the solver class you can have a look at [these](https://github.com/pytorch/examples) PyTorch examples.

# In[1]:


import numpy as np
import os
from random import choice
from string import ascii_uppercase
import torch
from torch.autograd import Variable
from yz.data_utils_harddisk import get_Cancer_datasets
from yz.solver import Solver
from yz.data_utils_harddisk import get_balanced_weights
from torchvision import models
import torch.nn as nn
import pandas as pd
from bayes_opt import BayesianOptimization
from tqdm import tqdm

csv_full_name = '~/dl4cvproject/data/train.csv'
img_folder_full_name = '~/dl4cvproject/data/train256'
csv_full_name = os.path.expanduser(csv_full_name)
img_folder_full_name = os.path.expanduser(img_folder_full_name)

csv_full_name_test = '~/dl4cvproject/data/test.csv'
img_folder_full_name_test = '~/dl4cvproject/data/test256'
csv_full_name_test = os.path.expanduser(csv_full_name_test)
img_folder_full_name_test = os.path.expanduser(img_folder_full_name_test)

#%matplotlib inline
#plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[2]:


train_data, val_data, test_data, train_label_list = get_Cancer_datasets(csv_full_name=csv_full_name,img_folder_full_name=img_folder_full_name)
test_X, csv_test = get_Cancer_datasets(csv_full_name=csv_full_name_test,img_folder_full_name=img_folder_full_name_test, mode='upload')
print("Train size: %i" % len(train_data))
print("Val size: %i" % len(val_data))
print("Test size: %i" % len(test_data))
print("upload size: {}".format(len(test_X)))


# In[3]:


if torch.cuda.is_available():
    print('Cuda available')
else:
    print('Cuda not available :(---(')


# In[4]:


def target(factor, batch_size, lr_const, lr_exp, weight_decay_const, weight_decay_exp, num_epochs):

    batch_size = int(batch_size)
    num_epochs = int(num_epochs)
    lr_const = int(lr_const)
    weight_decay_const = int(weight_decay_const)
    
    #training
    weights = get_balanced_weights(label_list=train_label_list, num_classes=14, factor=factor)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=20, shuffle=False, num_workers=8)
    
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 14)

    lr = lr_const * np.power(10, lr_exp)
    weigth_decay = weight_decay_const * np.power(10, weight_decay_exp)    
    solver = Solver(optim_args={"lr":lr, "weight_decay":weigth_decay})
    solver.train(model, train_loader, val_loader, log_nth=1, num_epochs=num_epochs)
    
    #compute local prediction acc
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=30, shuffle=False, num_workers=8)
    scores = []
    for inputs, target in tqdm(test_loader):
        inputs, targets = Variable(inputs), Variable(target)
        if torch.cuda.is_available:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        scores.extend((preds == targets).data.cpu().numpy())
        
    test_acc = np.mean(scores)
    print(test_acc)
    
    ## generate submission file: submissions/res18_acc_randomsuffix.csv
    if test_acc >= 0.32:
        
        file_name = 'submissions/res18_' + '{:.5f}'.format(test_acc) + '_' + ''.join(choice(ascii_uppercase) for i in range(7)) + '.csv'
        print(file_name)
        
        try:
            del csv_test['age']
        except KeyError as e:
            print(e)
        try:
            del csv_test['gender']
        except KeyError as e:
            print(e)
        try:
            del csv_test['view_position']
        except KeyError as e:
            print(e)
        try:
            del csv_test['image_name']
        except KeyError as e:
            print(e)
        try:
            del csv_test['detected']
        except KeyError as e:
            print(e)

        pred_set = set()
        detected = []
        for i in tqdm(range(len(test_X))):
            tmp_pred_list = [0] * 14
            inputs = test_X[i]
            inputs = Variable(inputs.unsqueeze(0))
            if torch.cuda.is_available:
                inputs = inputs.cuda()
            for trial in range(1):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                int_label = preds.data.cpu().numpy().tolist()[0]
                tmp_pred_list[int_label] += 1

            pred = tmp_pred_list.index(max(tmp_pred_list))
            str_pred = 'class_' + str(pred + 1)
            detected.append(str_pred)

        csv_test['detected'] = pd.Series(detected)
        csv_test.to_csv(file_name, index=False)
    
    return test_acc

    


# ## Bayesian Optimization

# In[5]:


bo = BayesianOptimization(target, {'factor':(0.5, 1), 'batch_size':(30, 80),
                                   'lr_const':(1, 10), 'lr_exp':(-3, -7),
                                   'weight_decay_const':(1, 10), 'weight_decay_exp':(-1, -6),
                                   'num_epochs':(1, 7)})


# In[ ]:


bo.maximize(init_points=2, n_iter=100000, acq='ucb', kappa=5)


