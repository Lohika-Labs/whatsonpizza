import json
import os

import operator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.models import load_model
from keras.preprocessing import image

import numpy as np

from .common import TAXONOMY_FILE, PROJECT_BASE

class TFClassifier(object):
    def __init__(self, model_path, labels_map):
        self.model = load_model(model_path)
        self.labels_map = labels_map

    def predict(self, image_bytes):
        image_bytes = np.expand_dims(image_bytes, axis=0)
        image_bytes = image_bytes / 255.0

        probs = self.model.predict(image_bytes)[0]

        result = {}
        for idx, prob in enumerate(probs):
            result[self.labels_map[str(idx)]] = prob

        sorted_results = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_results


class TFBackend(object):
    def __init__(self):
        self.classifier = TFClassifier(os.path.join(PROJECT_BASE, 'playground', 'classifier', 'keras', 'inception_v3.h5'), self.read_label_map())

    @staticmethod
    def read_label_map():
        label_map = {}
        taxonomy = json.loads(open(TAXONOMY_FILE, 'r').read())
        i = 0
        for ptype in taxonomy.get('pizza_types'):
            label_map[str(i)] = ptype.get('name')
            i += 1
        return label_map

    def tensorflow_predict_image(self, image_path):
        img = image.load_img(image_path,
                             target_size=(299, 299))
        img = image.img_to_array(img)
        results = self.classifier.predict(img)
        return results
