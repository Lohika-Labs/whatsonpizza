#-*- coding: utf-8 -*-
""" TensorFlow recognition module """

# stdlib
import json
import os
import operator
import time

# 3rd party
import numpy

from keras.models import load_model
from keras.preprocessing import image

from .common import PROJECT_BASE
from .logger import logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


MODEL_DIR = os.path.join(PROJECT_BASE, 'models', 'tensorflow')
MODEL = os.path.join(MODEL_DIR, 'inception_v3.h5')
MODEL_LABELS = os.path.join(MODEL_DIR, 'label_map.json')

class TFClassifier(object):  # pylint:disable=too-few-public-methods
    """ TensorFlow classifier """
    def __init__(self, model_path, labels_map):
        start = time.time()
        self.model = load_model(model_path)
        self.labels_map = labels_map
        logger.warning("TensorFlow model loaded in: %ss", round(time.time() - start, 2))

    def predict(self, image_bytes):
        """ Predict image using Tensor"""
        image_bytes = numpy.expand_dims(image_bytes, axis=0)
        image_bytes = image_bytes / 255.0

        probs = self.model.predict(image_bytes)[0]

        result = {}
        for idx, prob in enumerate(probs):
            result[self.labels_map[str(idx)]] = prob

        sorted_results = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_results


class TFBackend(object):
    """ TensorFlow backend implementation ."""
    def __init__(self):
        self.classifier = TFClassifier(MODEL, self.read_label_map())

    @staticmethod
    def read_label_map():
        """ Read label map (comes with trained model)"""
        taxonomy = json.loads(open(MODEL_LABELS, 'r').read())
        return taxonomy

    def tensorflow_predict_image(self, image_path):
        """ Predict category using TensorFlow """
        img = image.load_img(image_path,
                             target_size=(299, 299))
        img = image.img_to_array(img)
        results = self.classifier.predict(img)
        return results
