import glob
from distutils.dir_util import mkpath
from multiprocessing.pool import Pool

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from scipy import ndimage, misc
import os


def create_seq_generator():

    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((2, 5),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.ContrastNormalization((0.5, 2.0))
                        )
                    ]),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    return seq

def generate_image_seq(image, num_samples):
    images = []
    seq_generator = create_seq_generator()
    for i in range(0, num_samples-1):
        images.append(seq_generator.augment_image(image))

    return images


# image = ndimage.imread("/tmp/food-101-10/train/apple_pie/134.jpg")
#
# augm_images = generate_image_seq(image, 15)
# for ai in augm_images:
#     misc.imshow(ai)


if __name__ == '__main__':
    input_dir = "/mnt/data/lab/datasets/pizza/pizza labeled/large_pruned/train"
    output_dir = "/mnt/data/lab/datasets/pizza/pizza labeled/large_pruned/augmented_train"

    num_augmentations = 15

    image_dirs = glob.glob(input_dir + "/*")
    # category dir -> category files
    image_files = map(lambda dir: (dir, glob.glob(dir + "/*")), image_dirs)

    def process(image_files_item):
        cat_path = image_files_item[0]
        files = image_files_item[1]
        print(cat_path, files)
        cat_name = os.path.basename(cat_path)
        for file in files:
            filename = os.path.basename(file)
            name, ext = filename.split(".")
            image = ndimage.imread(file)
            output_cat_dir = output_dir + "/" + cat_name + "/"
            mkpath(output_cat_dir)
            counter = 0
            for aug_image in generate_image_seq(image, num_augmentations):
                aug_filename = output_cat_dir + name + "_" + str(counter) + "." + ext
                counter += 1
                print(aug_filename)
                misc.imsave(aug_filename, aug_image)

    pool = Pool(6)
    pool.map(process, image_files)

    # for cat_path, files in image_files:
    #     print(cat_path, files)
    #     cat_name = os.path.basename(cat_path)
    #     for file in files:
    #         filename = os.path.basename(file)
    #         name, ext =filename.split(".")
    #         image = ndimage.imread(file)
    #         output_cat_dir = output_dir + "/"+cat_name + "/"
    #         mkpath(output_cat_dir)
    #         counter = 0
    #         for aug_image in generate_image_seq(image, num_augmentations):
    #             aug_filename = output_cat_dir + name+"_"+str(counter) + "."+ext
    #             counter+=1
    #             print(aug_filename)
    #             misc.imsave(aug_filename, aug_image)


