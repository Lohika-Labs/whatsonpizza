import os

import numpy as np
from keras.applications import InceptionV3
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class PizzaDetector:
    def __init__(self, weights):
        self.model = InceptionV3()
        self.model.load_weights(weights)

    def is_pizza(self, image_bytes):
        image_bytes = np.expand_dims(image_bytes, axis=0)
        image_bytes = image_bytes / 255.0
        preds = self.model.predict(image_bytes)
        preds = decode_predictions(preds)[0][:3]

        is_found = len(list(filter(lambda x: x[1] == "pizza", preds))) == 1
        return is_found


if __name__ == '__main__':
     pizz_detector = PizzaDetector(weights="inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
     img = image.load_img("/Users/bturkynewych/WORK/Whatson_pizza/whatsonpizza/whatsonpizza_backend/testset/1.jpg",
                          target_size=(299, 299))
     img = image.img_to_array(img)

     print(pizz_detector.is_pizza(img))
#
#     pizza_dir = "/mnt/data/lab/datasets/food/food-101-10/omelette"
#     files = os.listdir(pizza_dir)
#     for file in files:
#         img = image.load_img(pizza_dir + "/" +file,
#                              target_size=(299, 299))
#         img = image.img_to_array(img)
#         print(pizz_detector.is_pizza(img))
#

