import json
import os
import cv2
from .crossentropy import *
import mxnet as mx
from collections import namedtuple
import numpy as np
import os.path

from PIL import Image

Image.Image.tostring = Image.Image.tobytes

from .common import PROJECT_BASE, TAXONOMY_FILE

MODEL = PROJECT_BASE + 'snapshots/model/resnet-152'
#MODEL = 'snapshots/multilabel-resnet-50'
#IMG_DIR = '/Users/bturkynewych/Downloads/Whatson_pizza/tmp/w1/dataset_18K/images' #'categorized/'
#CAT_DIR = "categories.txt"

CAT_NUM = 24
NDAR_ZEROS = 24
LABEL_SHP = 24


BCH_SZ = 1
CLS_NUM = 24


def loadmodel(modelname, n, dshapes, lshapes):
    sym, arg_params, aux_params = mx.model.load_checkpoint(modelname, n)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu())
    mod.bind(for_training=False, data_shapes=dshapes, label_shapes=lshapes)
    arg_params['prob_label'] = mx.nd.array([0])
    mod.set_params(arg_params, aux_params)
    return mod


def prepareNDArray(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (50, 50,))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return mx.nd.array(img)


def CM(true, pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for ind in range(CLS_NUM):
        true_l = true[ind].asnumpy().tolist()[0]
        pred_l = pred[ind].asnumpy().tolist()[0]
        if true_l == 1.0 and pred_l >= 0.5:
            tp += 1
        elif true_l == 1.0 and pred_l < 0.5:
            fn += 1
        elif true_l == 0.0 and pred_l >= 0.5:
            fp += 1
        else:
            tn += 1
    return float(tp), float(fp), float(tn), float(fn)


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

def get_cats():
    cats = []
    taxonomy = json.loads(open(TAXONOMY_FILE, 'r').read())
    i = 0
    for ptype in taxonomy.get('pizza_types'):
        cats.append(ptype.get('name'))
    return cats

def predict(img):
    #raw = open(CAT_DIR).read()
    #with open('categories.txt', 'r') as f:
    #    cats = [l.rstrip() for l in f]
    cats = get_cats()
    print (cats)
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    if os.path.exists(img):
        ndar = prepareNDArray(img)
        label_ph = mx.nd.zeros((1, NDAR_ZEROS))
        mod = loadmodel(MODEL, 2, dshapes=[('data', ndar.shape)], lshapes=[('softmax_label', (1, LABEL_SHP))])
        Batch = namedtuple('Batch', ['data', 'label'])
        mod.forward(Batch([ndar], [label_ph]))
        prob = mod.get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)
        a = np.argsort(prob)[::-1]
        '''
        temp_tp, temp_fp, temp_tn, temp_fn = CM(true, prob)
        tp += temp_tp
        fp += temp_fp
        tn += temp_tn
        fn += temp_fn
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        '''
        '''
        ing_inds = []
        for ind in range(CAT_NUM):
            if prob[ind] >= 0.5:
                ing_inds.append(ind)

        y = np.argsort(np.squeeze(prob))[::-1]
        '''
        print (a)
        return cats[a[0]], prob[a[0]]


#IMG = IMG_DIR + '/10002.jpg'# 'Sfincione/' + '4158' + '.jpg'
#print (os.path.exists(IMG))
#tup = 
#print (predict(IMG))
#print ("Label:", tup[0])
#print ("Accuracy:", tup[1])
