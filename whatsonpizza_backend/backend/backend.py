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
        for ptype, score in backend.mxnet_predict_image(image_path):
            results.append({'name': ptype, 'value': str(round(score, 2))})
        return results

    def tensorflow_analyze_image(self, image_path):
        """ Analyze image using TensorFlow """
        if not self.tensorflow:
            self.tensorflow = TFBackend()
        results = []
        logger.warning('Analyzing `%s` using TensorFlow', image_path)
        for name, probability in self.tensorflow.tensorflow_predict_image(image_path):
            results.append({'name': name, 'value': str(round(float(probability), 2))})
        return results
