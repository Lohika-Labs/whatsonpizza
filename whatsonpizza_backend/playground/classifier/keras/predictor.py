import json
import os

from keras.preprocessing import image

from playground.classifier.keras.TFClassifier import TFClassifier

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':

    # train_dir = "/mnt/data/lab/datasets/pizza/pizza labeled/small/augmented_train"
    # train_datagen = ImageDataGenerator()
    # generator = train_datagen.flow_from_directory(train_dir, batch_size=256)
    # label_map = generator.class_indices
    # label_map = {v: k for k, v in label_map.items()}
    # print(label_map)
    # json.dump(label_map, open("labels_map.json", "w"))

    curr_dir = os.path.dirname(os.path.realpath(__file__))
    labels_map = json.load(open(curr_dir + "/" + "labels_map.json", "r"))

    tf_classifier = TFClassifier("inception_v3.h5", labels_map)

    pizza_dir = "/mnt/data/lab/datasets/pizza/pizza labeled/small/validation/Sicilian/"
    files = os.listdir(pizza_dir)
    for file in files:
        img = image.load_img(pizza_dir + file,
                             target_size=(299, 299))
        img = image.img_to_array(img)

        results = tf_classifier.predict(img)
        print(file, results)
