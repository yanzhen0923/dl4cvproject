import imageio
import numpy as np
import scipy.misc
import os
import pandas as pd
from tqdm import tqdm
from skimage import exposure
from scipy import stats

def main():
    prefixTr = '/Users/yuminsun/dl4cvproject/data/train_'
    prefixTr256 = '/Users/yuminsun/dl4cvproject/data/train256'
    img_path = 'scan_000123.png'
    fullname = os.path.join(prefixTr, img_path)
    new_img = scipy.misc.imresize(imageio.imread(fullname), (256, 256))

    hist_img = exposure.equalize_hist(new_img)
    hist_img = 1 - hist_img
    
    img_hist = 'hist_000123.png'
    vmin, vmax = stats.scoreatpercentile(hist_img, (0.5, 99.5))
    img = np.clip(hist_img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin)
    
    img_minmax = 'minmax_000123.png'
    scipy.misc.imsave(os.path.join(prefixTr256, img_hist), hist_img)
    scipy.misc.imsave(os.path.join(prefixTr256, img_minmax), img)


if __name__ == '__main__':
    main()


