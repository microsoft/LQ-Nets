#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: learned_quantization.py

import tensorflow as tf
from tensorflow.contrib.framework import add_model_variable
from tensorflow.python.training import moving_averages
from tensorpack.models import *
from tensorpack.tfutils.tower import get_current_tower_context

MOVING_AVERAGES_FACTOR = 0.9
EPS = 0.0001
NORM_PPF_0_75 = 0.6745


@layer_register()
def QuantizedActiv(x, nbit=2):
    """
    Quantize activation.
    Args:
        x (tf.Tensor): a 4D tensor.
        nbit (int): number of bits of quantized activation. Defaults to 2.
    Returns:
        tf.Tensor with attribute `variables`.
    Variable Names:
    * ``basis``: basis of quantized activation.
    Note:
        About multi-GPU training: moving averages across GPUs are not aggregated.
        Batch statistics are computed by main training tower. This is consistent with most frameworks.
    """
    init_basis = [(NORM_PPF_0_75 * 2 / (2 ** nbit - 1)) * (2. ** i) for i in range(nbit)]
    init_basis = tf.constant_initializer(init_basis)
    bit_dims = [nbit, 1]
    num_levels = 2 ** nbit
    # initialize level multiplier
    init_level_multiplier = []
    for i in range(0, num_levels):
        level_multiplier_i = [0. for j in range(nbit)]
        level_number = i
        for j in range(nbit):
            level_multiplier_i[j] = float(level_number % 2)
            level_number = level_number // 2
        init_level_multiplier.append(level_multiplier_i)
    # initialize threshold multiplier
    init_thrs_multiplier = []
    for i in range(1, num_levels):
        thrs_multiplier_i = [0. for j in range(num_levels)]
        thrs_multiplier_i[i - 1] = 0.5
        thrs_multiplier_i[i] = 0.5
        init_thrs_multiplier.append(thrs_multiplier_i)

    with tf.variable_scope('ActivationQuantization'):
        basis = tf.get_variable(
            'basis', bit_dims, tf.float32,
            initializer=init_basis,
            trainable=False)

        ctx = get_current_tower_context()  # current tower context
        # calculate levels and sort
        level_codes = tf.constant(init_level_multiplier)
        levels = tf.matmul(level_codes, basis)
        levels, sort_id = tf.nn.top_k(tf.transpose(levels, [1, 0]), num_levels)
        levels = tf.reverse(levels, [-1])
        sort_id = tf.reverse(sort_id, [-1])
        levels = tf.transpose(levels, [1, 0])
        sort_id = tf.transpose(sort_id, [1, 0])
        # calculate threshold
        thrs_multiplier = tf.constant(init_thrs_multiplier)
        thrs = tf.matmul(thrs_multiplier, levels)
        # calculate output y and its binary code
        y = tf.zeros_like(x)  # output
        reshape_x = tf.reshape(x, [-1])
        zero_dims = tf.stack([tf.shape(reshape_x)[0], nbit])
        bits_y = tf.fill(zero_dims, 0.)
        zero_y = tf.zeros_like(x)
        zero_bits_y = tf.fill(zero_dims, 0.)
        for i in range(num_levels - 1):
            g = tf.greater(x, thrs[i])
            y = tf.where(g, zero_y + levels[i + 1], y)
            bits_y = tf.where(tf.reshape(g, [-1]), zero_bits_y + level_codes[sort_id[i + 1][0]], bits_y)
        # training
        if ctx.is_main_training_tower:
            BT = tf.matrix_transpose(bits_y)
            # calculate BTxB
            BTxB = []
            for i in range(nbit):
                for j in range(nbit):
                    BTxBij = tf.multiply(BT[i], BT[j])
                    BTxBij = tf.reduce_sum(BTxBij)
                    BTxB.append(BTxBij)
            BTxB = tf.reshape(tf.stack(values=BTxB), [nbit, nbit])
            BTxB_inv = tf.matrix_inverse(BTxB)
            # calculate BTxX
            BTxX = []
            for i in range(nbit):
                BTxXi0 = tf.multiply(BT[i], reshape_x)
                BTxXi0 = tf.reduce_sum(BTxXi0)
                BTxX.append(BTxXi0)
            BTxX = tf.reshape(tf.stack(values=BTxX), [nbit, 1])

            new_basis = tf.matmul(BTxB_inv, BTxX)  # calculate new basis
            # create moving averages op
            updata_moving_basis = moving_averages.assign_moving_average(
                basis, new_basis, MOVING_AVERAGES_FACTOR)
            add_model_variable(basis)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, updata_moving_basis)

            for i in range(nbit):
                tf.summary.scalar('basis%d' % i, new_basis[i][0])

        x_clip = tf.minimum(x, levels[num_levels - 1])  # gradient clip
        y = x_clip + tf.stop_gradient(-x_clip) + tf.stop_gradient(y)  # gradient: y=clip(x)
        y.variables = VariableHolder(basis=basis)
        return y


def QuantizedWeight(name, x, n, nbit=2):
    """
    Quantize weight.
    Args:
        x (tf.Tensor): a 4D tensor.
            Must have known number of channels, but can have other unknown dimensions.
        name (str): operator's name.
        n (int or double): variance of weight initialization.
        nbit (int): number of bits of quantized weight. Defaults to 2.
    Returns:
        tf.Tensor with attribute `variables`.
    Variable Names:
    * ``basis``: basis of quantized weight.
    Note:
        About multi-GPU training: moving averages across GPUs are not aggregated.
        Batch statistics are computed by main training tower. This is consistent with most frameworks.
    """
    num_filters = x.get_shape().as_list()[-1]
    init_basis = []
    base = NORM_PPF_0_75 * ((2. / n) ** 0.5) / (2 ** (nbit - 1))
    for j in range(nbit):
        init_basis.append([(2 ** j) * base for i in range(num_filters)])
    init_basis = tf.constant_initializer(init_basis)
    bit_dims = [nbit, num_filters]
    num_levels = 2 ** nbit
    delta = EPS
    # initialize level multiplier
    init_level_multiplier = []
    for i in range(num_levels):
        level_multiplier_i = [0. for j in range(nbit)]
        level_number = i
        for j in range(nbit):
            binary_code = level_number % 2
            if binary_code == 0:
                binary_code = -1
            level_multiplier_i[j] = float(binary_code)
            level_number = level_number // 2
        init_level_multiplier.append(level_multiplier_i)
    # initialize threshold multiplier
    init_thrs_multiplier = []
    for i in range(1, num_levels):
        thrs_multiplier_i = [0. for j in range(num_levels)]
        thrs_multiplier_i[i - 1] = 0.5
        thrs_multiplier_i[i] = 0.5
        init_thrs_multiplier.append(thrs_multiplier_i)

    with tf.variable_scope(name):
        basis = tf.get_variable(
            'basis', bit_dims, tf.float32,
            initializer=init_basis,
            trainable=False)
        level_codes = tf.constant(init_level_multiplier)
        thrs_multiplier = tf.constant(init_thrs_multiplier)
        sum_multiplier = tf.constant(1., shape=[1, tf.reshape(x, [-1, num_filters]).get_shape()[0]])
        sum_multiplier_basis = tf.constant(1., shape=[1, nbit])

        ctx = get_current_tower_context()  # current tower context
        # calculate levels and sort
        levels = tf.matmul(level_codes, basis)
        levels, sort_id = tf.nn.top_k(tf.transpose(levels, [1, 0]), num_levels)
        levels = tf.reverse(levels, [-1])
        sort_id = tf.reverse(sort_id, [-1])
        levels = tf.transpose(levels, [1, 0])
        sort_id = tf.transpose(sort_id, [1, 0])
        # calculate threshold
        thrs = tf.matmul(thrs_multiplier, levels)
        # calculate level codes per channel
        reshape_x = tf.reshape(x, [-1, num_filters])
        level_codes_channelwise_dims = tf.stack([num_levels * num_filters, nbit])
        level_codes_channelwise = tf.fill(level_codes_channelwise_dims, 0.)
        for i in range(num_levels):
            eq = tf.equal(sort_id, i)
            level_codes_channelwise = tf.where(tf.reshape(eq, [-1]), level_codes_channelwise + level_codes[i], level_codes_channelwise)
        level_codes_channelwise = tf.reshape(level_codes_channelwise, [num_levels, num_filters, nbit])
        # calculate output y and its binary code
        y = tf.zeros_like(x) + levels[0]  # output
        zero_dims = tf.stack([tf.shape(reshape_x)[0] * num_filters, nbit])
        bits_y = tf.fill(zero_dims, -1.)
        zero_y = tf.zeros_like(x)
        zero_bits_y = tf.fill(zero_dims, 0.)
        zero_bits_y = tf.reshape(zero_bits_y, [-1, num_filters, nbit])
        for i in range(num_levels - 1):
            g = tf.greater(x, thrs[i])
            y = tf.where(g, zero_y + levels[i + 1], y)
            bits_y = tf.where(tf.reshape(g, [-1]), tf.reshape(zero_bits_y + level_codes_channelwise[i + 1], [-1, nbit]), bits_y)
        bits_y = tf.reshape(bits_y, [-1, num_filters, nbit])
        # training
        if ctx.is_main_training_tower:
            BT = tf.transpose(bits_y, [2, 0, 1])
            # calculate BTxB
            BTxB = []
            for i in range(nbit):
                for j in range(nbit):
                    BTxBij = tf.multiply(BT[i], BT[j])
                    BTxBij = tf.matmul(sum_multiplier, BTxBij)
                    if i == j:
                        mat_one = tf.ones([1, num_filters])
                        BTxBij = BTxBij + (delta * mat_one)  # + E
                    BTxB.append(BTxBij)
            BTxB = tf.reshape(tf.stack(values=BTxB), [nbit, nbit, num_filters])
            # calculate inverse of BTxB
            if nbit > 2:
                BTxB_transpose = tf.transpose(BTxB, [2, 0, 1])
                BTxB_inv = tf.matrix_inverse(BTxB_transpose)
                BTxB_inv = tf.transpose(BTxB_inv, [1, 2, 0])
            elif nbit == 2:
                det = tf.multiply(BTxB[0][0], BTxB[1][1]) - tf.multiply(BTxB[0][1], BTxB[1][0])
                inv = []
                inv.append(BTxB[1][1] / det)
                inv.append(-BTxB[0][1] / det)
                inv.append(-BTxB[1][0] / det)
                inv.append(BTxB[0][0] / det)
                BTxB_inv = tf.reshape(tf.stack(values=inv), [nbit, nbit, num_filters])
            elif nbit == 1:
                BTxB_inv = tf.reciprocal(BTxB)
            # calculate BTxX
            BTxX = []
            for i in range(nbit):
                BTxXi0 = tf.multiply(BT[i], reshape_x)
                BTxXi0 = tf.matmul(sum_multiplier, BTxXi0)
                BTxX.append(BTxXi0)
            BTxX = tf.reshape(tf.stack(values=BTxX), [nbit, num_filters])
            BTxX = BTxX + (delta * basis)  # + basis
            # calculate new basis
            new_basis = []
            for i in range(nbit):
                new_basis_i = tf.multiply(BTxB_inv[i], BTxX)
                new_basis_i = tf.matmul(sum_multiplier_basis, new_basis_i)
                new_basis.append(new_basis_i)
            new_basis = tf.reshape(tf.stack(values=new_basis), [nbit, num_filters])
            # create moving averages op
            updata_moving_basis = moving_averages.assign_moving_average(
                basis, new_basis, MOVING_AVERAGES_FACTOR)
            add_model_variable(basis)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, updata_moving_basis)

        y = x + tf.stop_gradient(-x) + tf.stop_gradient(y)  # gradient: y=x
        y.variables = VariableHolder(basis=basis)
        return y


@layer_register()
def Conv2DQuant(x, out_channel, kernel_shape,
                padding='SAME', stride=1,
                W_init=None, b_init=None,
                nl=tf.identity, split=1, use_bias=True,
                data_format='NHWC', is_quant=True, nbit=1, fc=False):
    """
    2D convolution on 4D inputs.
    Args:
        x (tf.Tensor): a 4D tensor.
            Must have known number of channels, but can have other unknown dimensions.
        out_channel (int): number of output channel.
        kernel_shape: (h, w) tuple or a int.
        stride: (h, w) tuple or a int.
        padding (str): 'valid' or 'same'. Case insensitive.
        split (int): Split channels as used in Alexnet. Defaults to 1 (no split).
        W_init: initializer for W. Defaults to `variance_scaling_initializer`.
        b_init: initializer for b. Defaults to zero.
        nl: a nonlinearity function.
        use_bias (bool): whether to use bias.
        data_format (str): 'NHWC' or 'NCHW'. Defaults to 'NHWC'.
        is_quant (bool): whether to quantize weight. Defaults to 'True'.
        nbit (int): number of bits of quantized weight. Defaults to 1.
        fc (bool): whether to convert Conv2D to FullyConnect. Defaults to 'False'.
    Returns:
        tf.Tensor named ``output`` with attribute `variables`.
    Variable Names:
    * ``W``: weights
    * ``b``: bias
    """
    n = kernel_shape * kernel_shape * out_channel
    in_shape = x.get_shape().as_list()
    channel_axis = 3 if data_format == 'NHWC' else 1
    in_channel = in_shape[channel_axis]
    assert in_channel is not None, "[Conv2DQuant] Input cannot have unknown channel!"
    assert in_channel % split == 0
    assert out_channel % split == 0

    if fc:
        x = tf.reshape(x, [-1, in_channel, 1, 1])

    kernel_shape = [kernel_shape, kernel_shape]
    padding = padding.upper()
    filter_shape = kernel_shape + [in_channel / split, out_channel]

    if data_format == 'NCHW':
        stride = [1, 1, stride, stride]
    else:
        stride = [1, stride, stride, 1]

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer()
    if b_init is None:
        b_init = tf.constant_initializer()

    W = tf.get_variable('W', filter_shape, initializer=W_init)

    kernel_in = W * 1
    tf.summary.scalar('weight', tf.reduce_mean(tf.abs(W)))
    if is_quant:
        quantized_weight = QuantizedWeight('weight_quant', kernel_in, n, nbit=nbit)
    else:
        quantized_weight = kernel_in

    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=b_init)

    if split == 1:
        conv = tf.nn.conv2d(x, quantized_weight, stride, padding, data_format=data_format)
    else:
        inputs = tf.split(x, split, channel_axis)
        kernels = tf.split(quantized_weight, split, 3)
        outputs = [tf.nn.conv2d(i, k, stride, padding, data_format=data_format)
                   for i, k in zip(inputs, kernels)]
        conv = tf.concat(outputs, channel_axis)

    ret = nl(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')
    ret.variables = VariableHolder(W=W)
    if use_bias:
        ret.variables.b = b
    if fc:
        ret = tf.reshape(ret, [-1, out_channel])
    return ret


@layer_register(log_shape=False, use_scope=None)
def BNReLUQuant(x):
    """
    A shorthand of BatchNormalization + ReLU + QuantizedActiv.
    """
    x = BatchNorm('bn', x)
    x = tf.nn.relu(x)
    x = QuantizedActiv('quant', x)
    return x


def getBNReLUQuant(x, name=None):
    """
    A shorthand of BatchNormalization + ReLU + QuantizedActiv.
    """
    x = BatchNorm('bn', x)
    x = tf.nn.relu(x, name=name)
    x = QuantizedActiv('quant', x)
    return x


def getfcBNReLUQuant(x, name=None):
    """
    A shorthand of BatchNormalization + ReLU + QuantizedActiv after FullyConnect.
    """
    x = BatchNorm('bn', x, data_format='NHWC', use_scale=False, use_bias=False)
    x = tf.nn.relu(x, name=name)
    x = QuantizedActiv('quant', x)
    return x


def getfcBNReLU(x, name=None):
    """
    A shorthand of BatchNormalization + ReLU after FullyConnect.
    """
    x = BatchNorm('bn', x, data_format='NHWC', use_scale=False, use_bias=False)
    x = tf.nn.relu(x, name=name)
    return x
