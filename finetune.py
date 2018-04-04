import mxnet as mx
import os, sys
import logging

head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

num_classes = 10
batch_per_gpu = 128
num_gpus = 2
batch_size = batch_per_gpu * num_gpus


def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urlretrieve(url, filename)


def get_iterators(batch_size, data_shape=(3, 50, 50)):
    train = mx.io.ImageRecordIter(
        path_imgrec='./categorized-train.rec',
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        shuffle=True,
        rand_crop=True,
        rand_mirro=True)
    val = mx.io.ImageRecordIter(
        path_imgrec='./categorized-val_train.rec',
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        rand_crop=True,
        rand_mirro=True)
    return train, val


def get_model(prefix, epoch):
    download(prefix + '-symbol.json')
    download(prefix + '-%04d.params' % (epoch,))


def get_finetune(symbol, arg_params, num_classes, layer_name="flatten0"):
    all_layers = symbol.get_internals()
    net = all_layers[layer_name + '_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})
    return net, new_args


def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=symbol, context=devs)
    mod.fit(train, val,
            num_epoch=8,
            arg_params=arg_params,
            aux_params=aux_params,
            allow_missing=True,
            batch_end_callback=mx.callback.Speedometer(batch_size, 10),
            kvstore='device',
            optimizer='sgd',
            optimizer_params={'learning_rate': 0.01},
            initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
            eval_metric='acc')
    metric = mx.metric.Accuracy()
    return mod.score(val, metric)


get_model('http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152', 0)
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', 0)

(train, val) = get_iterators(batch_size)
(new_sym, new_args) = get_finetune(sym, arg_params, num_classes)
mod_score = fit(new_sym, new_args, aux_params, train, val, batch_size, num_gpus)
print mod_score
