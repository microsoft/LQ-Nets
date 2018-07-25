#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar10-vgg-small.py

import argparse
import os

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.utils.gpu import get_nr_gpu

from learned_quantization import *

BATCH_SIZE = 100
NUM_UNITS = None
WEIGHT_DECAY = 5e-4


class Model(ModelDesc):

    def __init__(self, qw=1, qa=0):
        super(Model, self).__init__()
        self.qw = qw
        self.qa = qa

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 128.0
        assert tf.test.is_gpu_available()
        image = tf.transpose(image, [0, 3, 1, 2])

        with argscope([Conv2DQuant, MaxPooling, BatchNorm], data_format='NCHW'), \
             argscope(Conv2DQuant, nl=tf.identity, use_bias=False, kernel_shape=3,
                      W_init=variance_scaling_initializer(mode='FAN_IN'),
                      nbit=self.qw, is_quant=True if self.qw > 0 else False):
            l = Conv2DQuant('conv0', image, 128, nl=BNReLU, is_quant=False)
            if self.qa > 0:
                l = QuantizedActiv('quant1', l, self.qa)
            l = Conv2DQuant('conv1', l, 128)
            # 32

            l = MaxPooling('pool2', l, shape=2, stride=2, padding='VALID')
            l = BNReLU('bn2', l)
            if self.qa > 0:
                l = QuantizedActiv('quant2', l, self.qa)
            l = Conv2DQuant('conv2', l, 256, nl=BNReLU)
            if self.qa > 0:
                l = QuantizedActiv('quant3', l, self.qa)
            l = Conv2DQuant('conv3', l, 256)
            # 16

            l = MaxPooling('pool4', l, shape=2, stride=2, padding='VALID')
            l = BNReLU('bn4', l)
            if self.qa > 0:
                l = QuantizedActiv('quant4', l, self.qa)
            l = Conv2DQuant('conv4', l, 512, nl=BNReLU)
            if self.qa > 0:
                l = QuantizedActiv('quant5', l, self.qa)
            l = Conv2DQuant('conv5', l, 512)
            # 8

            l = MaxPooling('pool6', l, shape=2, stride=2, padding='VALID')
            l = BNReLU('bn6', l)
            # 4

        logits = FullyConnected('linear', l, out_dim=10, nl=tf.identity)
        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label)
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        wd_cost = tf.multiply(WEIGHT_DECAY, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))  # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.02, summary=True)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--qw',
                        help='quantization weight',
                        type=int, default=1)
    parser.add_argument('--qa',
                        help='quantization activation',
                        type=int, default=0)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--logdir', help='identify of logdir',
                        type=str, default='cifar10-vgg-small')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.set_logger_dir(
        os.path.join('train_log', args.logdir))

    dataset_train = get_data('train')
    dataset_test = get_data('test')

    config = TrainConfig(
        model=Model(qw=args.qw, qa=args.qa),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError()]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.02), (80, 0.002), (160, 0.0002), (300, 0.00002)])
        ],
        max_epoch=400,
        nr_tower=max(get_nr_gpu(), 1),
        session_init=SaverRestore(args.load) if args.load else None
    )
    SyncMultiGPUTrainerParameterServer(config).train()
