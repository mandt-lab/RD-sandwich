# Prepare images for training; go through imgs in input_glob, select those >= 512x512,
# and downsample them by a random factor to remove possible compression artifacts from JPEG.
# Example: python prepare_imgs.py 'src_imgs/*.jpg' my_downsampled_imgs/

# Yibo Yang, 2021

import os
import numpy as np
import sys
import glob
from PIL import Image
from multiprocessing import Pool


def process_img(img_path, output_dir, min_hw=(512, 512), avg_downsample_factor=0.8):
    img = Image.open(img_path)
    if img.height < min_hw[0] or img.width < min_hw[1]:
        return -1
    x = np.array(img)
    if len(x.shape) != 3 or x.shape[2] != 3:
        return -2  # non-color image

    # e.g., downsample_factor ~ Uniform[0.6, 1) if avg_downsample_factor=0.8
    downsample_factor = np.random.uniform(low=1 - 2 * (1 - avg_downsample_factor), high=1.)
    downsampled_size = (round(img.width * downsample_factor), round(img.height * downsample_factor))  # PIL needs (w,h)
    out_img = img.resize(downsampled_size, Image.BICUBIC)

    img_name = os.path.splitext(os.path.basename(img_path))[0]  # xyz.jpeg -> xyz
    out_path = os.path.join(output_dir, f'{img_name}-downsampled.png')
    out_img.save(out_path)
    return out_path


if __name__ == '__main__':
    input_glob = sys.argv[1]
    output_dir = sys.argv[2]
    img_paths = glob.glob(input_glob)


    def fun(img_path):
        return process_img(img_path, output_dir, min_hw=(512, 512), avg_downsample_factor=0.8)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    with Pool() as p:
        if tqdm:
            res = list(tqdm(p.imap(fun, img_paths), total=len(img_paths)))
        else:
            res = list(p.imap(fun, img_paths))
