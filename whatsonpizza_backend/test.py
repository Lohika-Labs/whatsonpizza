#!/usr/bin/env python3

import os
from glob import glob
from backend import Backend
from backend.common import DATASET_BASE

b = Backend()
images = glob("./testset/*.jpg")

for img in images:
    print (b.tensorflow_analyze_image(img))

for img in images:
    print (b.mxnet_analyze_image(img))
