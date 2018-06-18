#-*- coding: utf-8 -*-
""" MXNet recognition module """

import os
import json

from collections import namedtuple, OrderedDict

import cv2
import mxnet as mx
import numpy as np


from .common import PROJECT_BASE
from .logger import logger


MODEL_DIR = os.path.join(PROJECT_BASE, 'models', 'mxnet')
MODEL = os.path.join(MODEL_DIR, 'Inception-100')
MODEL_LABELS = os.path.join(MODEL_DIR, 'label_map.json')
MODEL_SYNSET = os.path.join(MODEL_DIR, 'synset.txt')


class MXNetBackend(object):
    """ MXNet classifier """
    def __init__(self):
        self.batch = namedtuple('Batch', ['data'])
        sym, arg_params, aux_params = mx.model.load_checkpoint(MODEL, 100)
        self.mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data', (1, 3, 299, 299))],
                      label_shapes=self.mod._label_shapes)  # pylint:disable=protected-access
        self.mod.set_params(arg_params, aux_params, allow_missing=True)

    @staticmethod
    def get_cats():
        """ Return list of categories """
        categories = []
        taxonomy = open(MODEL_SYNSET, 'r')
        #for _, category in OrderedDict(taxonomy).items():
         #   categories.append(category)
        for l in taxonomy:
            categories.append(l.rstrip())
        return categories

    @staticmethod
    def get_image(fname):
        """ Create image object from file """
        img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)  # pylint:disable=no-member
        if img is None:
            return None
        img = cv2.resize(img, (299, 299))  # pylint:disable=no-member
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :]
        return img

    def mxnet_predict_image(self, fname):
        """ Predict category using MXNet """
        img = self.get_image(fname)
        self.mod.forward(self.batch([mx.nd.array(img)]))
        prob = self.mod.get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)
        sorted = np.argsort(prob)[::-1]
        cat = self.get_cats()
        logger.warning('MXNet categories: %s', cat)
        results = []
        for idx in sorted[0:5]:
            tup = (cat[idx], prob[idx])
            results.append(tup)
        return results
