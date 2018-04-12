import mxnet as mx
import cv2
import numpy as np
import json
# define a simple data batch
from collections import namedtuple
from PIL import Image

from .common import TAXONOMY_FILE, PROJECT_BASE

MODEL = PROJECT_BASE + 'snapshots/resnet-50'


Batch = namedtuple('Batch', ['data'])
Image.Image.tostring = Image.Image.tobytes

#MODEL = 'resnet-50'
ctx = mx.cpu()

sym, arg_params, aux_params = mx.model.load_checkpoint(MODEL, 50)
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)


def get_cats():
    cats = []
    taxonomy = json.loads(open(TAXONOMY_FILE, 'r').read())
    for ptype in taxonomy.get('pizza_types'):
        cats.append(ptype.get('name'))
    return cats


def get_image(fname):
    #fname = mx.test_utils.download(url)
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img


def predict(fname):
    img = get_image(fname)
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    acc = np.argsort(prob)[::-1]
    cat = get_cats()
    ALL = []
    for i in range(9):
        tup = (cat[i], prob[i])
        ALL.append(tup)
    #print (ALL)
    return ALL
    #print ALL


#predict('http://images.pizza33.ua/products/product/POCpLYdcgVA34bcde4pK8JEjSWITKbtk.jpg')
