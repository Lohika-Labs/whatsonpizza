from __future__ import absolute_import

from .common import PROJECT_BASE
from .demo import predict
from .tfclassifier import TFBackend


class Backend(object):
    def __init__(self):
        self.tf = None

    def mxnet_analyze_image(self, image_path):
        results = []
        print ('Analyzing using MXNet ', image_path)
        for ptype, score in predict(image_path):
            results.append({'name': ptype, 'value': round(score, 2)})
        return results

    def tensorflow_analyze_image(self, image_path):
        if not self.tf:
            self.tf = TFBackend()
        results = []
        print ('Analyzing using TensorFlow ', image_path)
        for a, b in self.tf.tensorflow_predict_image(image_path):
            results.append({'name': a, 'value': round(float(b), 2)})
        return results