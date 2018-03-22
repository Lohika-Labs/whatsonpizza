from __future__ import absolute_import

from .common import PROJECT_BASE
from .demo import predict

class Backend(object):
    def __init__(self):
        print (PROJECT_BASE)

    def mxnet_analyze_image(self, image_path):
        print (image_path)
        return predict(image_path)
