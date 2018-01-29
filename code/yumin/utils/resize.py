import imageio
import scipy.misc
import os
import pandas as pd
from tqdm import tqdm
from skimage import exposure

def main():
	#csvTe_name = '/Users/yuminsun/dl4cvproject/data/test.csv'
	#test = pd.read_csv(csvTe_name)
	#prefixTe = '/Users/yuminsun/dl4cvproject/data/test_'
	#prefixTe256 = '/Users/yuminsun/dl4cvproject/data/test256'
	csvTr_name = '/Users/yuminsun/dl4cvproject/data/train.csv'
	train = pd.read_csv(csvTr_name)
	prefixTr = '/Users/yuminsun/dl4cvproject/data/train_'
	prefixTr256 = '/Users/yuminsun/dl4cvproject/data/train256'
	for img_path in tqdm(train['image_name'].values):
		fullname = os.path.join(prefixTr, img_path)
		new_img = scipy.misc.imresize(imageio.imread(fullname), (256, 256))
		hist_img = exposure.equalize_hist(new_img)
		scipy.misc.imsave(os.path.join(prefixTr256, img_path), hist_img)
    
    #p=Augmentor.Pipeline(prefixTr256)
    #p.rotate90(probability=0.3)
    #p.rotate270(probability=0.3)
    #p.flip_left_right(probability=0.3)
    #p.flip_top_bottom(probability=0.3)
    #p.sample(1000)
        

if __name__ == '__main__':
    main()

