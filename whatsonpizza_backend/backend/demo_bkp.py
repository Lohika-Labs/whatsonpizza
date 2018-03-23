import cv2
import mxnet as mx
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from random import randint
import os.path

from PIL import Image

Image.Image.tostring = Image.Image.tobytes

from .common import PROJECT_BASE
from .crossentropy import *

MODEL = PROJECT_BASE + 'snapshots/multilabel-resnet-50'
IMG_DIR = PROJECT_BASE + 'dataset_18K/images/'
CAT_DIR = PROJECT_BASE + "dataset_18K/cats_single.txt"
ctx = mx.cpu()


def gen_id():
    return randint(1, 18728)


def loadmodel(modelname, n, dshapes, lshapes):
    sym, arg_params, aux_params = mx.model.load_checkpoint(modelname, n)
    mod = mx.mod.Module(symbol=sym, context=ctx)
    mod.bind(for_training=False, data_shapes=dshapes, label_shapes=lshapes)
    arg_params['prob_label'] = mx.nd.array([0])
    mod.set_params(arg_params, aux_params)
    return mod


def prepareNDArray(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224,))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return mx.nd.array(img)


def predict(img):
    raw = open(CAT_DIR).read()
    cats = raw.split('\n')
    if os.path.exists(img):
        ndar = prepareNDArray(img)
        label_ph = mx.nd.zeros((1, 52))
        mod = loadmodel(MODEL, 2, dshapes=[('data', ndar.shape)], lshapes=[('softmax_label', (1, 52))])
        Batch = namedtuple('Batch', ['data', 'label'])
        mod.forward(Batch([ndar], [label_ph]))
        prob = mod.get_outputs()[0][0]
        ing_inds = []
        for ind in range(52):
            if prob[ind] >= 0.5:
                ing_inds.append(ind)

        #for ind in ing_inds:
        #    print (cats[ind])

        #print (prob, np.squeeze(prob))
        y = np.argsort(np.squeeze(prob))[::-1]

        return cats[ing_inds[0]], y[0]

if __name__ == '__main__':
    IMG = IMG_DIR + str(randint(1, 18728)) + '.jpg'
    tup = predict(IMG)
    print ("Label:", tup[0])
    print ("Accuracy:", tup[1])
