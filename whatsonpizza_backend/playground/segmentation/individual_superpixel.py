# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from skimage import io




image_path="/home/vbartko/Downloads/whatsonpizza/dataset_1K/5_recipes/image_1082 .jpg"

image = cv2.imread(image_path)
segments = slic(img_as_float(image), n_segments = 100, sigma = 5)

# show the output of SLIC
fig = plt.figure("Superpixels")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
plt.axis("off")
plt.show()

# loop over the unique segment values
for (i, segVal) in enumerate(np.unique(segments)):
	# construct a mask for the segment
	print ("[x] inspecting segment %d" % (i))
	mask = np.zeros(image.shape[:2], dtype = "uint8")
	mask[segments == segVal] = 255

	# show the masked region
	cv2.imshow("Mask", mask)
	cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))
	cv2.waitKey(0)