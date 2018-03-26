#!/usr/bin/env python3

import os

from backend import Backend
from backend.common import DATASET_BASE

b = Backend()
for i in range(1, 10):
    print (b.mxnet_analyze_image(os.path.join(DATASET_BASE, 'images', '1000%s.jpg' % i)))

for i in range(1, 10):
    print (b.tensorflow_analyze_image(os.path.join(DATASET_BASE, 'images', '1000%s.jpg' % i)))
