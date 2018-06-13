import mxnet as mx
import cv2
import numpy as np
# define a simple data batch
from collections import namedtuple
from PIL import Image

Image.Image.tostring = Image.Image.tobytes

MODEL = 'Inception'
SYNSET = 'synset.txt'

def loadInceptionv3():
    sym, arg_params, aux_params = mx.model.load_checkpoint(MODEL, 50)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 299, 299))])
    mod.set_params(arg_params, aux_params, allow_missing=True)
    return mod


def loadCategories():
    synsetfile = open(SYNSET, 'r')
    synsets = []
    for l in synsetfile:
        synsets.append(l.rstrip())
    return synsets


def prepareNDArray(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299,))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return mx.nd.array(img)


def predict(filename, model, categories, n):
    array = prepareNDArray(filename)
    Batch = namedtuple('Batch', ['data'])
    model.forward(Batch([array]))
    prob = model.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    sortedprobindex = np.argsort(prob)[::-1]
    topn = []
    for i in sortedprobindex[0:n]:
        topn.append(('{:f}'.format(prob[i]), categories[i]))
    return topn


def detect(filename):
    model = loadInceptionv3()
    cats = loadCategories()
    topn = predict(filename, model, cats, 5)
    return topn

det = detect('data/orig/Bolognese/18587.jpg')
print det