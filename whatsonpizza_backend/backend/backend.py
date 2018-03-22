from __future__ import absolute_import

from .common import PROJECT_BASE
from .demo import predict

class Backend(object):
    def __init__(self):
        pass

    def mxnet_analyze_image(self, image_path):
        print ('Analyzing ', image_path)
        ptype, score = predict(image_path)
        return {'name': ptype, 'value': score/100}
