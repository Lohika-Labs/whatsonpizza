import mxnet as mx
ctx = mx.cpu()
import json
from crossentropy import *

TEST_DIR = 'data_18K/test_data.lst'
IMG_DIR = 'dataset_18K/images/'
# batch_size = 64
batch_size = 1
num_classes = 52
MODEL = 'snapshots/multilabel-resnet-50'
epochs = 13
# epochs = 1
num_examples = 1898


def loadmodel(modelname, n, iter):
    sym, arg_params, aux_params = mx.model.load_checkpoint(modelname, n)
    mod = mx.mod.Module(symbol=sym, context=ctx)
    mod.bind(for_training=False, data_shapes=iter.provide_data, label_shapes=iter.provide_label)
    arg_params['prob_label'] = mx.nd.array([0])
    mod.set_params(arg_params, aux_params)
    return mod


def CM(true, pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for ind in range(num_classes):
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


testiter = mx.image.ImageIter(
    batch_size=batch_size,
    data_shape=(3, 224, 224),
    label_width=num_classes,
    path_imglist=TEST_DIR,
    path_root=IMG_DIR,
    data_name='data',
    label_name='softmax_label',
    aug_list=mx.image.CreateAugmenter((3, 224, 224), resize=224, mean=True, std=True))

ALL = {}

for epoch in range(1, 2):

    testiter.reset()
    mod = loadmodel(MODEL, epoch, testiter)

    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    for batch in testiter:
        mod.forward(batch)
        pred = mod.get_outputs()[0][0]
        true = batch.label[0][0]
        temp_tp, temp_fp, temp_tn, temp_fn = CM(true, pred)
        tp += temp_tp
        fp += temp_fp
        tn += temp_tn
        fn += temp_fn

    metrics = {}

    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    F1 = 2 * (precision * recall)/(precision + recall)

    metrics["accuracy"] = accuracy
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["F1"] = F1
    ALL[epoch] = metrics

with open('results.json', 'w') as outfile:
    json.dumps(ALL, outfile)
