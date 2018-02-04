import os,sys
import numpy as np
import scipy.misc
import torchvision.transforms as transforms
import glob
import matplotlib.pyplot as plt

def data_augmentation():

    trsfmToPIL= transforms.Compose([transforms.ToPILImage()])
    trsfmToTensor = transforms.ToTensor()
    transform =transforms.Compose([transforms.ToPILImage(),
                                   transforms.ColorJitter(brightness=0.5),
                                   transforms.RandomCrop(224),
                                   transforms.Resize(256),
                                   ])
    prefix = '/Users/yuminsun/dl4cvproject/data/train256/output/final2'
    file ='/Users/yuminsun/dl4cvproject/data/train256/minmax_000123.png'

    img = scipy.misc.imread(file)
    # repeat channels
    one_channel_img = np.expand_dims(img, axis=2)
    # numpy H x W x C
    three_channel_img = np.repeat(one_channel_img, 3, axis=2)

    for i in range(10):
        aug_PIL = transform(three_channel_img)
        aug_np = np.array(aug_PIL)
        img_path = str(i)+'.png'
        print(img_path)
        scipy.misc.imsave(os.path.join(prefix,img_path),aug_PIL)

    return
