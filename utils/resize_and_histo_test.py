import imageio
import scipy.misc
import os
import pandas as pd
from tqdm import tqdm
from skimage import exposure

def main():
	csvTe_name = '/home/ubuntu/dl4cvproject/data/test.csv'
	test = pd.read_csv(csvTe_name)
	prefixTe = '/home/ubuntu/dl4cvproject/data/test_'
	prefixTe256 = '/home/ubuntu/dl4cvproject/data/test360'
	#csvTr_name = '/home/ubuntu/dl4cvproject/data/train.csv'
	#train = pd.read_csv(csvTr_name)
	#prefixTr = '/home/ubuntu/dl4cvproject/data/train_'
	#prefixTr256 = '/home/ubuntu/dl4cvproject/data/train360'
	for img_path in tqdm(test['image_name'].values):
		fullname = os.path.join(prefixTe, img_path)
		new_img = scipy.misc.imresize(imageio.imread(fullname), (360, 360))
		hist_img = exposure.equalize_hist(new_img)
		scipy.misc.imsave(os.path.join(prefixTe256, img_path), hist_img)

if __name__ == '__main__':
    main()

