#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: googlenet_model.py

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack.models import *
from tensorpack.tfutils.argscope import argscope, get_arg_scope

from learned_quantization import Conv2DQuant, getBNReLUQuant, QuantizedActiv


def inception_block(l, name, ch_1x1, ch_3x3, ch_5x5, is_last_block=False, is_last=False):
    data_format = get_arg_scope()['Conv2DQuant']['data_format']
    with tf.variable_scope(name):
        conv1x1 = Conv2DQuant('1x1', l, ch_1x1, 1, nl=getBNReLUQuant if not is_last_block else tf.identity)
        conv3x3_reduce = Conv2DQuant('3x3_reduce', l, ch_3x3, 1, nl=getBNReLUQuant)
        conv3x3 = Conv2DQuant('3x3', conv3x3_reduce, ch_3x3, 3, nl=getBNReLUQuant if not is_last_block else tf.identity)
        conv5x5_reduce = Conv2DQuant('5x5_reduce', l, ch_5x5, 1, nl=getBNReLUQuant)
        conv5x5 = Conv2DQuant('5x5', conv5x5_reduce, ch_5x5, 5, nl=getBNReLUQuant if not is_last_block else tf.identity)
        if is_last_block and not is_last:
            conv1x1 = MaxPooling('pool_1x1', conv1x1, shape=3, stride=2, padding='SAME')
            conv1x1 = BNReLU('conv1x1_bn', conv1x1)
            conv1x1 = QuantizedActiv('conv1x1_quant', conv1x1)
            conv3x3 = MaxPooling('pool_3x3', conv3x3, shape=3, stride=2, padding='SAME')
            conv3x3 = BNReLU('conv3x3_bn', conv3x3)
            conv3x3 = QuantizedActiv('conv3x3_quant', conv3x3)
            conv5x5 = MaxPooling('pool_5x5', conv5x5, shape=3, stride=2, padding='SAME')
            conv5x5 = BNReLU('conv5x5_bn', conv5x5)
            conv5x5 = QuantizedActiv('conv5x5_quant', conv5x5)
        l = tf.concat([
            conv1x1,
            conv3x3,
            conv5x5], 1 if data_format == 'NCHW' else 3, name='concat')
        if is_last:
            l = BNReLU('output_bn', l)
    return l


def googlenet_backbone(image, qw=1):
    with argscope(Conv2DQuant, nl=tf.identity, use_bias=False,
                  W_init=variance_scaling_initializer(mode='FAN_IN'),
                  data_format=get_arg_scope()['Conv2D']['data_format'],
                  nbit=qw,
                  is_quant=True if qw > 0 else False):
        logits = (LinearWrap(image)
                  .Conv2DQuant('conv1', 64, 7, stride=2, is_quant=False)
                  .MaxPooling('pool1', shape=3, stride=2, padding='SAME')
                  .BNReLUQuant('pool1/out')
                  .Conv2DQuant('conv2/3x3_reduce', 192, 1, nl=getBNReLUQuant)
                  .Conv2DQuant('conv2/3x3', 192, 3)
                  .MaxPooling('pool2', shape=3, stride=2, padding='SAME')
                  .BNReLUQuant('pool2/out')
                  .apply(inception_block, 'incpetion_3a', 96, 128, 32)
                  .apply(inception_block, 'incpetion_3b', 192, 192, 96, is_last_block=True)
                  .apply(inception_block, 'incpetion_4a', 256, 208, 48)
                  .apply(inception_block, 'incpetion_4b', 224, 224, 64)
                  .apply(inception_block, 'incpetion_4c', 192, 256, 64)
                  .apply(inception_block, 'incpetion_4d', 176, 288, 64)
                  .apply(inception_block, 'incpetion_4e', 384, 320, 128, is_last_block=True)
                  .apply(inception_block, 'incpetion_5a', 384, 320, 128)
                  .apply(inception_block, 'incpetion_5b', 512, 384, 128, is_last_block=True, is_last=True)
                  .GlobalAvgPooling('pool5')
                  .FullyConnected('linear', out_dim=1000, nl=tf.identity)())
    return logits
