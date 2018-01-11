import imageio
import scipy.misc
import os
import pandas as pd
from tqdm import tqdm


def main():
    csv_name = '/home/hpc/pr92no/ga42cih2/Projects/dl4cvproject/data/test.csv'
    train = pd.read_csv(csv_name)

    prefix = '/home/hpc/pr92no/ga42cih2/Projects/dl4cvproject/data/test_'
    prefix400 = '/home/hpc/pr92no/ga42cih2/Projects/dl4cvproject/data/test_400'

    for img_path in tqdm(train['image_name'].values):
        fullname = os.path.join(prefix, img_path)
        new_img = scipy.misc.imresize(imageio.imread(fullname), (400, 400))
        scipy.misc.imsave(os.path.join(prefix400, img_path), new_img)

if __name__ == '__main__':
    main()

