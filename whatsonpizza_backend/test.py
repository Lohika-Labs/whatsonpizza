#!/usr/bin/env python3

import os

from backend import Backend
from backend.common import DATASET_BASE

b = Backend()
print (b.mxnet_analyze_image(os.path.join(DATASET_BASE, 'images', '10002.jpg')))