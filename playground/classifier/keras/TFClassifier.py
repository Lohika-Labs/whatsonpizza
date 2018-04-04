import os

import operator

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.models import load_model
import numpy as np


class TFClassifier:
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
