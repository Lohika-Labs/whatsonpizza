#-*- coding: utf-8 -*-
""" Pizza detector module """

import json
import os

import numpy
from keras.applications import InceptionV3, imagenet_utils
from keras.preprocessing import image

from .common import PROJECT_BASE


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


MODEL_DIR = os.path.join(PROJECT_BASE, 'models', 'tensorflow')
MODEL = os.path.join(MODEL_DIR, 'inception_v3_detection.h5')
CLASS_INDEX = os.path.join(MODEL_DIR, 'imagenet_class_index.json')


class PizzaDetector(object):  # pylint:disable=too-few-public-methods
    """ Pizza object detection using Keras/Tensorflow """
    def __init__(self, model_path=None, class_index=None):
        if class_index and os.path.exists(class_index) and os.path.isfile(class_index):
            imagenet_utils.CLASS_INDEX = json.loads(open(class_index, 'r').read())
        self.model = InceptionV3(weights=model_path)

    def is_pizza(self, image_bytes):
        """ Check if image in buffer is actually a pizza """
        image_bytes = numpy.expand_dims(image_bytes, axis=0)
        image_bytes = image_bytes / 255.0
        preds = self.model.predict(image_bytes)
        preds = imagenet_utils.decode_predictions(preds)[0][:3]

        is_found = len(list(filter(lambda x: x[1] == "pizza", preds))) == 1
        return is_found


class PizzaDetectorWrapper(object):  # pylint:disable=too-few-public-methods
    """ Pizza detection wrapper """
    def __init__(self):
        self.pizza_detector = None

    def detect_pizza(self, image_path):
        """ Detect whether specified image file is a pizza """
        if not self.pizza_detector:
            self.pizza_detector = PizzaDetector(model_path=MODEL, class_index=CLASS_INDEX)
        img = image.load_img(image_path,
                             target_size=(299, 299))
        img = image.img_to_array(img)
        return self.pizza_detector.is_pizza(img)
