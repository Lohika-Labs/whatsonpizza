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
    mod = mx.mod.Module.load(prefix="Inception", epoch=epoch, context=devs)
    mod.bind(data_shapes=[('data', (1, 3, 229, 229))])
    mod.score(eval_data=test, eval_metric=['ce', 'acc'])
