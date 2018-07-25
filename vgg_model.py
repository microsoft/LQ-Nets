#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vgg_model.py

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack.models import *
from tensorpack.tfutils.argscope import argscope, get_arg_scope

from learned_quantization import Conv2DQuant, getBNReLUQuant, getfcBNReLUQuant, getfcBNReLU


def vgg_backbone(image, qw=1):
    with argscope(Conv2DQuant, nl=tf.identity, use_bias=False,
                  W_init=variance_scaling_initializer(mode='FAN_IN'),
                  data_format=get_arg_scope()['Conv2D']['data_format'],
                  nbit=qw):
        logits = (LinearWrap(image)
                  .Conv2DQuant('conv1', 96, 7, stride=2, nl=tf.nn.relu, is_quant=False)
                  .MaxPooling('pool1', shape=2, stride=2, padding='VALID')
                  # 56
                  .BNReLUQuant('bnquant2_0')
                  .Conv2DQuant('conv2_1', 256, 3, nl=getBNReLUQuant)
                  .Conv2DQuant('conv2_2', 256, 3, nl=getBNReLUQuant)
                  .Conv2DQuant('conv2_3', 256, 3)
                  .MaxPooling('pool2', shape=2, stride=2, padding='VALID')
                  # 28
                  .BNReLUQuant('bnquant3_0')
                  .Conv2DQuant('conv3_1', 512, 3, nl=getBNReLUQuant)
                  .Conv2DQuant('conv3_2', 512, 3, nl=getBNReLUQuant)
                  .Conv2DQuant('conv3_3', 512, 3)
                  .MaxPooling('pool3', shape=2, stride=2, padding='VALID')
                  # 14
                  .BNReLUQuant('bnquant4_0')
                  .Conv2DQuant('conv4_1', 512, 3, nl=getBNReLUQuant)
                  .Conv2DQuant('conv4_2', 512, 3, nl=getBNReLUQuant)
                  .Conv2DQuant('conv4_3', 512, 3)
                  .MaxPooling('pool4', shape=2, stride=2, padding='VALID')
                  # 7
                  .BNReLUQuant('bnquant5')
                  .Conv2DQuant('fc5', 4096, 7, nl=getfcBNReLUQuant, padding='VALID', use_bias=True)
                  .Conv2DQuant('fc6', 4096, 1, nl=getfcBNReLU, padding='VALID', use_bias=True)
                  .FullyConnected('fc7', out_dim=1000, nl=tf.identity, W_init=variance_scaling_initializer(mode='FAN_IN'))())
    return logits
