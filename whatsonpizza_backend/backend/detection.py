#-*- coding: utf-8 -*-
""" Pizza detector module """

import os

import numpy
from keras.applications import InceptionV3
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image


from .common import PROJECT_BASE
from .logger import logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


MODEL_DIR = os.path.join(PROJECT_BASE, 'models', 'tensorflow')
MODEL = os.path.join(MODEL_DIR, 'inception_v3_detection.h5')


class PizzaDetector(object):  # pylint:disable=too-few-public-methods
    """ Pizza object detection using Keras/Tensorflow """
    def __init__(self, model_path):
        self.model = InceptionV3()
        self.model.load_weights(model_path)

    def is_pizza(self, image_bytes):
        """ Check if image in buffer is actually a pizza """
        image_bytes = numpy.expand_dims(image_bytes, axis=0)
        image_bytes = image_bytes / 255.0
        preds = self.model.predict(image_bytes)
        preds = decode_predictions(preds)[0][:3]

        is_found = len(list(filter(lambda x: x[1] == "pizza", preds))) == 1
        return is_found

class PizzaDetectorWrapper(object):  # pylint:disable=too-few-public-methods
    """ Pizza detection wrapper """
    def __init__(self):
        self.pizza_detector = None

    def detect_pizza(self, image_path):
        """ Detect whether specified image file is a pizza """
        if not self.pizza_detector:
            logger.warning('%s - %s', os.path.exists(MODEL), 'PDW')
            self.pizza_detector = PizzaDetector(MODEL)
        img = image.load_img(image_path,
                             target_size=(299, 299))
        img = image.img_to_array(img)
        return self.pizza_detector.is_pizza(img)
