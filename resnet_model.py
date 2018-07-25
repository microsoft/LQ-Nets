#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack.models import *
from tensorpack.tfutils.argscope import argscope, get_arg_scope

from learned_quantization import Conv2DQuant, QuantizedActiv


def resnet_shortcut(l, n_out, stride, nl=tf.identity, block_type='B'):
    data_format = get_arg_scope()['Conv2DQuant']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format == 'NCHW' else 3]
    if n_in != n_out:  # change dimension when channel is not the same
        if block_type == 'B':
            return Conv2DQuant('convshortcut', l, n_out, 1, stride=stride, nl=nl)
        else:
            l = AvgPooling('poolshortcut', l, stride, stride, padding='VALID')
            if data_format == 'NCHW':
                paddings = [[0, 0], [0, n_out - n_in], [0, 0], [0, 0]]
            else:
                paddings = [[0, 0], [0, 0], [0, 0], [0, n_out - n_in]]
            l = tf.pad(l, paddings, 'CONSTANT')
            return l
    else:
        return l


def apply_preactivation(l, preact, block_func):
    if preact == 'bnrelu':
        shortcut = l  # preserve identity mapping
        l = BNReLU('preact', l)
        l = QuantizedActiv('quant1', l)
    elif preact == 'first':
        if block_func == 'basic':
            shortcut = l
            l = QuantizedActiv('quant1', l)
        else:
            l = QuantizedActiv('quant1', l)
            shortcut = l
    else:
        shortcut = l
    return l, shortcut


def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name: BatchNorm('bn', x, gamma_init=tf.zeros_initializer())
    else:
        return lambda x, name: BatchNorm('bn', x)


def preresnet_basicblock(l, ch_out, stride, preact, block_type='B'):
    l, shortcut = apply_preactivation(l, preact, 'basic')
    l = Conv2DQuant('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = QuantizedActiv('quant2', l)
    l = Conv2DQuant('conv2', l, ch_out, 3, nl=get_bn(zero_init=False))
    return l + resnet_shortcut(shortcut, ch_out, stride, nl=get_bn(zero_init=False), block_type=block_type)


def preresnet_bottleneck(l, ch_out, stride, preact, block_type='A'):
    # stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact, 'basic')
    l = Conv2DQuant('conv1', l, ch_out, 1, nl=BNReLU)
    l = QuantizedActiv('quant2', l)
    l = Conv2DQuant('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = QuantizedActiv('quant3', l)
    l = Conv2DQuant('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=False))
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False), block_type=block_type)


def preresnet_group(l, name, block_func, features, count, stride, is_last=False):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                if i == 0 and stride == 1:
                    preact = 'first'
                elif i == 0:
                    preact = 'no_preact'
                else:
                    preact = 'bnrelu'
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               preact, block_type='B')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
        if not is_last:
            l = QuantizedActiv('quant_last', l)
    return l


def preresnet_group_typeA(l, name, block_func, features, count, stride, is_last=False):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                if i == 0 and stride == 1:
                    preact = 'first'
                elif i == 0:
                    preact = 'first'
                else:
                    preact = 'bnrelu'
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               preact, block_type='A')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l


def resnet_basicblock(l, ch_out, stride):
    shortcut = l
    l = Conv2DQuant('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2DQuant('conv2', l, ch_out, 3, nl=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out, stride, nl=get_bn(zero_init=False))


def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2DQuant('conv1', l, ch_out, 1, stride=stride if stride_first else 1, nl=BNReLU)
    l = Conv2DQuant('conv2', l, ch_out, 3, stride=1 if stride_first else stride, nl=BNReLU)
    l = Conv2DQuant('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False))


def resnet_group(l, name, block_func, features, count, stride, is_last=False):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
                # end of each block need an activation
                l = tf.nn.relu(l)
    return l


def resnet_backbone(image, num_blocks, group_func, block_func, qw=1):
    with argscope(Conv2DQuant, nl=tf.identity, use_bias=False,
                  W_init=variance_scaling_initializer(mode='FAN_OUT'),
                  data_format=get_arg_scope()['Conv2D']['data_format'],
                  nbit=qw):
        logits = (LinearWrap(image)
                  .Conv2DQuant('conv0', 64, 7, stride=2, nl=BNReLU, is_quant=False)
                  .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                  .apply(group_func, 'group0', block_func, 64, num_blocks[0], 1)
                  .apply(group_func, 'group1', block_func, 128, num_blocks[1], 2)
                  .apply(group_func, 'group2', block_func, 256, num_blocks[2], 2)
                  .apply(group_func, 'group3', block_func, 512, num_blocks[3], 2, is_last=True)
                  .GlobalAvgPooling('gap')
                  .FullyConnected('linear', 1000, nl=tf.identity)())
    return logits
