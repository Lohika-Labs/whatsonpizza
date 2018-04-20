#-*- coding: utf-8 -*-
""" Backend main file """
from __future__ import absolute_import

from .mxclassifier import MXNetBackend
from .tfclassifier import TFBackend
from .logger import  logger


class Backend(object):
    """ Recognition backend """
    def __init__(self):
        self.tensorflow = None

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
