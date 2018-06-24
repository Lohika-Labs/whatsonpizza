import mxnet as mx
import logging

head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

batch_per_gpu = 16
num_gpus = 2
batch_size = batch_per_gpu * num_gpus

test = mx.io.ImageRecordIter(
    path_imgrec='./data-test.rec',
    data_name='data',
    label_name='softmax_label',
    batch_size=batch_size,
    data_shape=(3, 299, 299),
    rand_crop=False,
    rand_mirro=False)

for epoch in range(1, 100):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    sym, arg_params, aux_params = mx.model.load_checkpoint("Inception", epoch)
    mod = mx.mod.Module(symbol=sym, context=devs, label_names=['softmax_label', ])
    mod.bind(for_training=False,
             data_shapes=test.provide_data,
             label_shapes=test.provide_label)
    mod.set_params(arg_params, aux_params)
    mod.score(eval_data=test, eval_metric=['acc', 'ec'])
