#-*- coding: utf-8 -*-
""" Backend main file """

import time

from .detection import PizzaDetectorWrapper
from .mxclassifier import MXNetBackend
from .tfclassifier import TFBackend
from .logger import  logger


def measure(func, *args, **kwargs):
    """ Wrap function and measure its execution time """
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return elapsed, result


class Backend(object):
    """ Recognition backend """
    def __init__(self):
        self.tensorflow = None
        self.pizza_detector = PizzaDetectorWrapper()

    @staticmethod
    def mxnet_analyze_image(image_path):
        """ Analyze image using MXNet"""
        results = []
        logger.warning('Analyzing `%s` using MXNet', image_path)
        backend = MXNetBackend()
        for name, probability in backend.mxnet_predict_image(image_path):
            probability = round(probability, 2)
            if probability == 0:
                continue
            results.append({'name': name, 'value': str(probability)})
        return results

    def tensorflow_analyze_image(self, image_path):
        """ Analyze image using TensorFlow """
        if not self.tensorflow:
            self.tensorflow = TFBackend()
        results = []
        logger.warning('Analyzing `%s` using TensorFlow', image_path)
        for name, probability in self.tensorflow.tensorflow_predict_image(image_path):
            probability = round(float(probability), 2)
            if probability == 0:
                continue
            results.append({'name': name, 'value': str(probability)})
        return results

    def analyze_image(self, image_path):
        """ Analyze image using both MXNet and Tensorflow"""
        data = {"status": "ok", "status_message": ""}
        detector_elapsed, detector_status = measure(self.pizza_detector.detect_pizza, image_path)
        if not detector_status:
            data = {"status": "error", "status_message": "Not a pizza",
                    "profiler": {"detector": str(round(detector_elapsed, 3))}}
        else:
            mxnet_elapsed, data['mxnet'] = measure(self.mxnet_analyze_image, image_path)
            tensorflow_elapsed, data['tensorflow'] = measure(self.tensorflow_analyze_image,
                                                             image_path)
            data['profiler'] = {'detector': str(round(detector_elapsed, 3)),
                                'mxnet': str(round(mxnet_elapsed, 3)),
                                'tensorflow': str(round(tensorflow_elapsed, 3))}
        return data
