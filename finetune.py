import mxnet as mx
import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

num_classes = 10
batch_per_gpu = 16
num_gpus = 2
batch_size = batch_per_gpu * num_gpus


def get_iterators(batch_size, data_shape=(3, 299, 299)):
    train = mx.io.ImageRecordIter(
        path_imgrec='./data-train.rec',
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        shuffle=True,
        rand_crop=True,
        rand_mirro=True)
    val = mx.io.ImageRecordIter(
        path_imgrec='./data-val.rec',
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        rand_crop=True,
        rand_mirro=True)
    return train, val


def do_finetune(symbol, arg_params):
    all_layers = symbol.get_internals()
    net = all_layers["flatten_output"]
    net = mx.symbol.Pooling(data=net, pool_type='max', stride=(1,1), kernel=(299, 299))
    net = mx.symbol.Dropout(data=net, p=0.7, name='dp', mode='always')
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})
    return net, new_args


def fit(symbol, arg_params, aux_params, train, val):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=symbol, context=devs)
    mod.fit(train, val,
            num_epoch=50,
            arg_params=arg_params,
            aux_params=aux_params,
            allow_missing=True,
            batch_end_callback=mx.callback.Speedometer(batch_size, 1),
            epoch_end_callback=mx.callback.do_checkpoint("Inception", 50),
            kvstore='device',
            optimizer='sgd',
            optimizer_params={'learning_rate': 0.01},
            initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
            eval_metric='ce')
    metric = mx.metric.create('ce')
    return mod.score(val, metric)


sym, arg_params, aux_params = mx.model.load_checkpoint('Inception-BN', 00)
(train, val) = get_iterators(batch_size)
(new_sym, new_args) = do_finetune(sym, arg_params)
mod_score = fit(new_sym, new_args, aux_params, train, val)
print (mod_score)
