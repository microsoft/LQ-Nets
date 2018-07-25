#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: alexnet_model.py

import tensorflow as tf
from tensorpack.models import *
from tensorpack.tfutils.argscope import argscope, get_arg_scope

from learned_quantization import Conv2DQuant, getBNReLUQuant, getfcBNReLU, getfcBNReLUQuant


def alexnet_backbone(image, qw=1):
    with argscope(Conv2DQuant, nl=tf.identity, use_bias=False,
                  W_init=tf.random_normal_initializer(stddev=0.01),
                  data_format=get_arg_scope()['Conv2D']['data_format'],
                  nbit=qw):
        logits = (LinearWrap(image)
                  .Conv2DQuant('conv1', 96, 11, stride=4, is_quant=False, padding='VALID')
                  .MaxPooling('pool1', shape=3, stride=2, padding='VALID')
                  .BNReLUQuant('bnquant2')
                  .Conv2DQuant('conv2', 256, 5)
                  .MaxPooling('pool2', shape=3, stride=2, padding='VALID')
                  .BNReLUQuant('bnquant3')
                  .Conv2DQuant('conv3', 384, 3, nl=getBNReLUQuant)
                  .Conv2DQuant('conv4', 384, 3, nl=getBNReLUQuant)
                  .Conv2DQuant('conv5', 256, 3)
                  .MaxPooling('pool5', shape=3, stride=2, padding='VALID')
                  .BNReLUQuant('bnquant6')
                  .Conv2DQuant('fc6', 4096, 6, nl=getfcBNReLUQuant, padding='VALID', W_init=tf.random_normal_initializer(stddev=0.005), use_bias=True)
                  .Conv2DQuant('fc7', 4096, 1, nl=getfcBNReLU, padding='VALID', W_init=tf.random_normal_initializer(stddev=0.005), use_bias=True)
                  .FullyConnected('fc8', out_dim=1000, nl=tf.identity, W_init=tf.random_normal_initializer(stddev=0.01))())
    return logits
