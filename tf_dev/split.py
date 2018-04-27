import glob
import os
import shutil
from distutils.dir_util import mkpath
from random import shuffle

import itertools

input_dir = "/mnt/data/lab/datasets/pizza/pizza labeled/large_pruned/orig"

out_dir = "/mnt/data/lab/datasets/pizza/pizza labeled/large_pruned"
train_dir = out_dir + "/train"
test_dir = out_dir + "/test"
validation_dir = out_dir + "/validation"

MIN_SAMPLES = 25

# sums up to ONE !!!!
splits_ratio = [0.75, 0.25, 0]


def prepare(input_dir, min_samples):
    image_dirs = glob.glob(input_dir + "/*")

    # category dir -> category files
    image_files = map(lambda dir: (dir, glob.glob(dir + "/*")), image_dirs)
    # remove categories with less than MIN_SAMPLES images
    image_files = filter(lambda dir_files: len(dir_files[1]) >= min_samples, image_files)
    # shuffle within each category
    map(lambda dir_files: (dir_files[0], shuffle(dir_files[1])), image_files)
    # take MIN_SAMPLES from each category
    map(lambda dir_files: (dir_files[0], dir_files[1][0:min_samples]), image_files)

    image_files_nested = map(lambda dir_files: (dir_files[1]), image_files)
    image_files = list(itertools.chain.from_iterable(image_files_nested))
    shuffle(image_files)

    return image_files


def get_all_splits(splits_ratio, image_files):
    train_idx = int(splits_ratio[0] * len(image_files))
    train_split = image_files[:train_idx]

    test_idx = int((splits_ratio[0] + splits_ratio[1]) * len(image_files))
    test_split = image_files[train_idx:test_idx]

   # validation_idx = int(splits_ratio[2] * len(image_files))
 #   validation_split = image_files[-validation_idx:]

    return train_split, test_split, None #validation_split


def copy_images(split, output_path):
    for image in split:
        image_dir = os.path.basename(os.path.dirname(image))
        full_image_dir = output_path + "/" + image_dir
        mkpath(full_image_dir)
        shutil.copy2(image, full_image_dir)
        print(full_image_dir)


if __name__ == '__main__':
    image_files = prepare(input_dir, MIN_SAMPLES)
    train_split, test_split, validation_split = get_all_splits(splits_ratio, image_files)

    copy_images(train_split, train_dir)
    copy_images(test_split, test_dir)
    #copy_images(validation_split, validation_dir)
