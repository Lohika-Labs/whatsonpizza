# import the necessary packages
import os

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse

#
# image_path="/home/vbartko/Downloads/whatsonpizza/dataset_1K/5_recipes/image_543.jpg"
#
# # load the image and convert it to a floating point data type
# image = img_as_float(io.imread(image_path))
#
# # loop over the number of segments
# for numSegments in (100, 200, 50,20,20):
#     # apply SLIC and extract (approximately) the supplied number
#     # of segments
#     segments = slic(image, n_segments=numSegments, sigma=5)
#
#     # show the output of SLIC
#     fig = plt.figure("Superpixels -- %d segments" % (numSegments))
#     ax = fig.add_subplot(1, 1, 1)
#     ax.imshow(mark_boundaries(image, segments))
#     plt.axis("off")
#
# # show the plots
# plt.show()


import glob

input_dir = "/home/vbartko/Downloads/whatsonpizza/dataset_1K/5_recipes"
output_dir = "/home/vbartko/Downloads/whatsonpizza/dataset_1K/superpixels"

img_files  = glob.glob(input_dir+"/*.jpg")
for img_file in img_files:
    print(img_file)
    image = img_as_float(io.imread(img_file))

    for numSegments in ([100]):
        segments = slic(image, n_segments=numSegments, sigma=5)
        base, ext = os.path.basename(img_file).split(".")
        boundries = mark_boundaries(image, segments)
        io.imsave(output_dir+"/"+base+"_"+str(numSegments)+"_.jpg",  boundries)


    # show the plots
    plt.show()
