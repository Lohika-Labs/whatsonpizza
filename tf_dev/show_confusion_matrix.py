import itertools
import json

from keras.preprocessing import image
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt

from playground.classifier.keras.TFClassifier import TFClassifier


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    y_true = ["qwerty","b", "b", "b", "c", "a", "qwerty"]
    y_pred = ["qwerty","b", "b", "a", "a", "c", "c"]
    cf = confusion_matrix(y_pred, y_true)
    print(cf)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cf, classes=["a","b", "c", "qwerty"],
                          title='Confusion matrix, without normalization')

    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cf, classes=["a","b", "c"], normalize=True,
    #                       title='Normalized confusion matrix')

    #create true labels
    #classify
    #build confusion matrix

    import os
    import glob


    curr_dir = os.path.dirname(os.path.realpath(__file__))
    labels_map = json.load(open(curr_dir + "/" + "labels_map.json", "r"))

    tf_classifier = TFClassifier("inception_v3.h5", labels_map)

    pizza_dir =  "/mnt/data/lab/datasets/pizza/pizza labeled/large_pruned/test"
    files = os.listdir(pizza_dir)

    pizza_true = []
    pizza_predicted = []


    for file in glob.glob(pizza_dir+"/*/*"):
        dir_name = os.path.basename(os.path.dirname(file))

        img = image.load_img(file,
                             target_size=(299, 299))
        img = image.img_to_array(img)

        results = tf_classifier.predict(img)
        pred_label =  results[0][0]
        print(dir_name, pred_label)

        pizza_true.append(dir_name)
        pizza_predicted.append(pred_label)


    cf = confusion_matrix(pizza_predicted, pizza_true)
    print(cf)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cf, classes=sorted(set(pizza_true)),
                          title='Confusion matrix, without normalization')

    plt.show()
