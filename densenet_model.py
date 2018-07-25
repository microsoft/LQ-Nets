#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: densenet_model.py

import math

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack.models import *
from tensorpack.tfutils.argscope import argscope, get_arg_scope

from learned_quantization import Conv2DQuant, QuantizedActiv

GROWTH_RATE = 32
REDUCTION = 0.5


def add_layer(name, l):
    shape = l.get_shape().as_list()
    in_channel = shape[1]
    with tf.variable_scope(name) as scope:
        c = Conv2DQuant('conv1x1', l, 4 * GROWTH_RATE, 1)
        c = BNReLU('bnrelu_2', c)
        c = QuantizedActiv('quant2', c)
        c = Conv2DQuant('conv3x3', c, GROWTH_RATE, 3)
        c = BNReLU('bnrelu_3', c)
        c = QuantizedActiv('quant3', c)
        l = tf.concat([c, l], 1)
    return l


def add_transition(name, l):
    shape = l.get_shape().as_list()
    in_channel = shape[1]
    out_channel = math.floor(in_channel * REDUCTION)
    with tf.variable_scope(name) as scope:
        l = Conv2DQuant('conv1', l, out_channel, 1, stride=1, use_bias=False)
        l = AvgPooling('pool', l, 2)
    return l


def add_dense_block(l, name, N, last=False, first=False):
    with tf.variable_scope(name) as scope:
        if first:
            l = BNReLU('first', l)
            l = QuantizedActiv('quant_first', l)
        for i in range(N):
            l = add_layer('dense_layer.{}'.format(i), l)
        if not last:
            l = add_transition('transition', l)
    return l


def densenet_backbone(image, qw=1):
    with argscope(Conv2DQuant, nl=tf.identity, use_bias=False,
                  W_init=variance_scaling_initializer(mode='FAN_IN'),
                  data_format=get_arg_scope()['Conv2D']['data_format'],
                  nbit=qw,
                  is_quant=True if qw > 0 else False):
        logits = (LinearWrap(image)
                  .Conv2DQuant('conv1', 2 * GROWTH_RATE, 7, stride=2, nl=BNReLU, is_quant=False)
                  .MaxPooling('pool1', shape=3, stride=2, padding='SAME')
                  # 56
                  .apply(add_dense_block, 'block0', 6)
                  # 28
                  .apply(add_dense_block, 'block1', 12)
                  # 14
                  .apply(add_dense_block, 'block2', 24)
                  # 7
                  .apply(add_dense_block, 'block3', 16, last=True)
                  .BNReLU('bnrelu_last')
                  .GlobalAvgPooling('gap')
                  .FullyConnected('linear', out_dim=1000, nl=tf.identity, W_init=variance_scaling_initializer(mode='FAN_IN'))())
    return logits
