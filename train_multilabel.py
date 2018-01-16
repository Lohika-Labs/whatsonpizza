import argparse
import logging
import sys
import os
import json
import mxnet as mx
import numpy as np
import dotenv
import ipdb
from crossentropy import *

dotenv.load_dotenv(dotenv.find_dotenv())

sys.path.insert(0, "./settings")
sys.path.insert(0, "../")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)


def get_fine_tune_model(sym, arg_params, num_classes, layer_name):

    all_layers = sym.get_internals()
    mnet = all_layers[layer_name + '_output']
    mnet = mx.symbol.FullyConnected(data=mnet, num_hidden=num_classes, name='fc')
    mnet = mx.symbol.sigmoid(data=mnet, name='sig')
    mnet = mx.symbol.Custom(data=mnet, name='softmax', op_type='CrossEntropyLoss')

    new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
    return (mnet, new_args)


def multi_factor_scheduler(begin_epoch, epoch_size, step=[5, 10], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None

def train_model(params):
    # pylint: disable=C0321,C0326
    train = mx.image.ImageIter(
        batch_size   = params.batch_size,
        data_shape   = params.image_shape,
        label_width  = params.num_classes,
        path_imglist = params.data_train,
        path_root    = params.image_train,
        part_index   = params.kv.rank,
        num_parts    = params.kv.num_workers,
        shuffle      = True,
        data_name    = 'data',
        label_name   = 'softmax_label',
        aug_list     = mx.image.CreateAugmenter(
            params.image_shape,
            resize = params.image_shape[-1],
            rand_crop = True,
            rand_mirror = True,
            mean = True,
            std = True))

    val = mx.image.ImageIter(
        batch_size   = params.batch_size,
        data_shape   = params.image_shape,
        label_width  = params.num_classes,
        path_imglist = params.data_val,
        path_root    = params.image_val,
        part_index   = params.kv.rank,
        num_parts    = params.kv.num_workers,
        data_name    = 'data',
        label_name   = 'softmax_label',
        aug_list     = mx.image.CreateAugmenter(
            params.image_shape,
            resize = params.image_shape[-1],
            mean = True,
            std = True))
    # pylint: enable=C0321,C0326

    kvstore = mx.kvstore.create(params.kv_store)

    prefix = params.model
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, params.epoch)

    (new_sym, new_args) = get_fine_tune_model(sym, arg_params, params.num_classes, 'flatten0')

    epoch_size = max(int(params.num_examples / params.batch_size / kvstore.num_workers), 1)
    lr_scheduler = multi_factor_scheduler(params.epoch, epoch_size)

    optimizer_params = {
        'learning_rate': params.lr,
        'momentum': params.mom,
        'wd': params.wd,
        'lr_scheduler': lr_scheduler
    }
    initializer = mx.init.Xavier(
        rnd_type='gaussian', factor_type="in", magnitude=2)

    if params.gpus:
        devs = [mx.gpu(int(dev_id)) for dev_id in params.gpus.split(',')]
    else:
        devs = mx.cpu()

    model = mx.mod.Module(context=devs, symbol=new_sym)

    checkpoint = mx.callback.do_checkpoint(
        params.save_result + params.save_name)

    def acc(label, pred, label_width=params.num_classes):
        return float(
            (label == np.round(pred)).sum()) / label_width / pred.shape[0]

    def loss(label, pred):
        eps = 1e-6
        loss_all = 0
        for k, _ in enumerate(pred):
            loss = 0
            loss -= label[k] * np.log(pred[k] + eps) + (1. - label[k]) * np.log(1. + eps - pred[k])
            loss_all += np.sum(loss)
        loss_all = float(loss_all) / float(len(pred) + eps)
        return loss_all


    eval_metric = list()
    eval_metric.append(mx.metric.np(acc))
    eval_metric.append(mx.metric.np(loss))

    model.fit(train,
              begin_epoch=params.epoch,
              num_epoch=params.num_epoch,
              eval_data=val,
              eval_metric=eval_metric,
              validation_metric=eval_metric,
              kvstore=kvstore,
              optimizer='sgd',
              optimizer_params=optimizer_params,
              arg_params=new_args,
              aux_params=aux_params,
              initializer=initializer,
              allow_missing=True,
              batch_end_callback=mx.callback.Speedometer(params.batch_size, 20),
              epoch_end_callback=checkpoint)

if __name__ == '__main__':
    # pylint: disable=C0326,C0301
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--model',         type=str,   default=os.environ['model'], required=False)
    parser.add_argument('--gpus',          type=str,   default=os.environ['gpus'])
    parser.add_argument('--batch-size',    type=int,   default=int(os.environ['batch-size']))
    parser.add_argument('--epoch',         type=int,   default=int(os.environ['epoch']))
    parser.add_argument('--image-shape',   type=str,   default=os.environ['image-shape'])
    parser.add_argument('--data-train',    type=str,   default=os.environ['data-train'])
    parser.add_argument('--image-train',   type=str,   default=os.environ['image-train'])
    parser.add_argument('--data-val',      type=str,   default=os.environ['data-val'])
    parser.add_argument('--image-val',     type=str,   default=os.environ['image-val'])
    parser.add_argument('--num-classes',   type=int,   default=int(os.environ['num-classes']))
    parser.add_argument('--lr',            type=float, default=float(os.environ['lr']))
    parser.add_argument('--num-epoch',     type=int,   default=int(os.environ['num-epoch']))
    parser.add_argument('--kv-store',      type=str,   default=os.environ['kv-store'], help='the kvstore type')
    parser.add_argument('--save-result',   type=str,   default=os.environ['save-result'], help='the save path')
    parser.add_argument('--num-examples',  type=int,   default=int(os.environ['num-examples']))
    parser.add_argument('--mom',           type=float, default=float(os.environ['mom']), help='momentum for sgd')
    parser.add_argument('--wd',            type=float, default=float(os.environ['wd']), help='weight decay for sgd')
    parser.add_argument('--save-name',     type=str,   default=os.environ['save-name'], help='the save name of model')
    args = parser.parse_args()
    # pylint: enable=C0326,C0301

    # logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)

    args.kv = mx.kvstore.create(args.kv_store)
    args.image_shape = tuple(np.fromstring(os.environ['image-shape'], dtype=np.int, sep=','))

    if not os.path.exists(args.save_result):
        os.mkdir(args.save_result)

    hdlr = logging.FileHandler(args.save_result+ '/train.log')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    args_str = json.dumps(vars(args), indent=2, sort_keys=True, default=lambda o: o.__dict__).replace('\"', '') # pretty print model args
    logging.info("Training params: {}".format(args_str))
    train_model(args)
