import mxnet as mx
import logging

head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

num_classes = 10
batch_per_gpu = 16
num_gpus = 2
batch_size = batch_per_gpu * num_gpus

'''
Creating two iterators (for training and validation data set parts)
Data shape that used for iteration = 3 * 299 * 299 where:
3 - dimension (RGB)
299 * 299 - width and height of images.

Params:
batch size - batch per gpu (16) * number of gpu (2)
Data shape = 3 * 299 * 299
'''
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


'''
Main transfer learning method.
Fetching all layers of current model (Inception-BN)
Finding layer called "flatten_output"
Then adding activation + dropout + fully connected layers
At the end added softmax output
Relu as activation function
Dropout with always mode (not only while training), 0.7 - dropout fraction
return all arguments of pre-trained model and architecture as 'net'.

Params:
symbol - configuration of network
arg_params - network weights
'''
def do_finetune(symbol, arg_params):
    all_layers = symbol.get_internals()
    net = all_layers["flatten_output"]
    net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    net = mx.symbol.Dropout(data=net, p=0.7, name='dp', mode='always')
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})
    return net, new_args


'''
Training method
Initialize number of contexts to train on (2 gpus)
Creating a metrics (accuracy and cross-entropy)
Training new weights with 100 epoch
Optimizer - SGD (with learning rate 0.01, momentum - 0.9, weight decay - 0.0005)

Params:
symbol - configuration of network
arg_params - network weights
aux_params - network auxiliary states
train - training iterator
val - validation iterator
'''
def fit(symbol, arg_params, aux_params, train, val):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=symbol, context=devs)
    metrics = mx.metric.create(['ce', 'acc'])
    mod.fit(train, val,
            num_epoch=100,
            arg_params=arg_params,
            aux_params=aux_params,
            allow_missing=True,
            # batch_end_callback=mx.callback.Speedometer(batch_size, 1),
            epoch_end_callback=mx.callback.do_checkpoint("Inception", 1),
            kvstore='device',
            optimizer='sgd',
            optimizer_params={'learning_rate': 0.01, 'wd': 0.0005, 'momentum': 0.9},
            initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
            eval_metric=metrics,
            validation_metric=metrics)


sym, arg_params, aux_params = mx.model.load_checkpoint('Inception-BN', 00)
(train, val) = get_iterators(batch_size)
(new_sym, new_args) = do_finetune(sym, arg_params)
fit(new_sym, new_args, aux_params, train, val)
