import mxnet as mx
import cv2
import numpy as np
# define a simple data batch
from collections import namedtuple
from PIL import Image

Batch = namedtuple('Batch', ['data'])
Image.Image.tostring = Image.Image.tobytes

MODEL = 'model/resnet-152'
ctx = mx.cpu()

sym, arg_params, aux_params = mx.model.load_checkpoint(MODEL, 0)
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)


def get_image(url):
    # download and show the image
    fname = mx.test_utils.download(url)
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img


def predict(url):
    img = get_image(url)
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    #for i in a[0:5]:
    print('probability=%f' % (prob[a[0]]))


predict('http://images.pizza33.ua/products/product/POCpLYdcgVA34bcde4pK8JEjSWITKbtk.jpg')