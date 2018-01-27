import imageio
import scipy.misc
import os
import pandas as pd
from tqdm import tqdm
from skimage import exposure

print('imported')

def main():
	csvTr_name = '~/dl4cvproject/data/train.csv'
	train = pd.read_csv(csvTr_name)
	prefixTr = '~/dl4cvproject/data/train_'
	prefixTr256 = '~/dl4cvproject/data/train256'
	for img_path in tqdm(train['image_name'].values):
		fullname = os.path.join(os.path.expanduser(prefixTr), img_path)
		new_img = scipy.misc.imresize(imageio.imread(fullname), (256, 256))
		hist_img = exposure.equalize_hist(new_img)
		scipy.misc.imsave(os.path.join(os.path.expanduser(prefixTr256), img_path), hist_img)

if __name__ == '__main__':
    main()

