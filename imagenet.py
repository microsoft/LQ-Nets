#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet.py

import argparse
import os

from tensorpack import logger, QueueInput
from tensorpack.callbacks import *
from tensorpack.dataflow import FakeData
from tensorpack.models import *
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.train import AutoResumeTrainConfig, SyncMultiGPUTrainerReplicated, launch_train_with_config
from tensorpack.utils.gpu import get_nr_gpu

import imagenet_utils
from alexnet_model import alexnet_backbone
from densenet_model import densenet_backbone
from googlenet_model import googlenet_backbone
from imagenet_utils import (
    fbresnet_augmentor, normal_augmentor, get_imagenet_dataflow, ImageNetModel,
    eval_on_ILSVRC12)
from learned_quantization import QuantizedActiv
from resnet_model import (
    preresnet_group, preresnet_group_typeA, preresnet_basicblock, preresnet_bottleneck,
    resnet_group, resnet_basicblock, resnet_bottleneck, resnet_backbone)
from vgg_model import vgg_backbone

TOTAL_BATCH_SIZE = 256


class Model(ImageNetModel):
    def __init__(self, depth, data_format='NCHW', mode='resnet', wd=5e-5, qw=1, qa=2, learning_rate=0.1, data_aug=True):
        super(Model, self).__init__(data_format, wd, learning_rate, data_aug, double_iter=True if TOTAL_BATCH_SIZE == 128 else False)

        self.mode = mode
        self.qw = qw
        self.qa = qa
        if mode == 'vgg' or mode == 'alexnet' or mode == 'googlenet' or mode == 'densenet':
            return
        basicblock = preresnet_basicblock if mode == 'preact' else resnet_basicblock
        bottleneck = {
            'resnet': resnet_bottleneck,
            'preact': preresnet_bottleneck,
            'preact_typeA': preresnet_bottleneck}[mode]
        self.num_blocks, self.block_func = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[depth]

    def get_logits(self, image):
        with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format), \
             argscope([QuantizedActiv], nbit=self.qa):
            if self.mode == 'vgg':
                return vgg_backbone(image, self.qw)
            elif self.mode == 'alexnet':
                return alexnet_backbone(image, self.qw)
            elif self.mode == 'googlenet':
                return googlenet_backbone(image, self.qw)
            elif self.mode == 'densenet':
                return densenet_backbone(image, self.qw)
            else:
                if self.mode == 'preact':
                    group_func = preresnet_group
                elif self.mode == 'preact_typeA':
                    group_func = preresnet_group_typeA
                else:
                    group_func = resnet_group
                return resnet_backbone(
                    image, self.num_blocks,
                    group_func, self.block_func, self.qw)


def get_data(name, batch, data_aug=True):
    isTrain = name == 'train'
    augmentors = fbresnet_augmentor(isTrain) if data_aug \
        else normal_augmentor(isTrain)
    return get_imagenet_dataflow(
        args.data, name, batch, augmentors)


def get_config(model, fake=False, data_aug=True):
    nr_tower = max(get_nr_gpu(), 1)
    batch = TOTAL_BATCH_SIZE // nr_tower

    if fake:
        logger.info("For benchmark, batch size is fixed to 64 per tower.")
        dataset_train = FakeData(
            [[64, 224, 224, 3], [64]], 1000, random=False, dtype='uint8')
        callbacks = []
    else:
        logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
        dataset_train = get_data('train', batch, data_aug)
        dataset_val = get_data('val', batch, data_aug)
        callbacks = [
            ModelSaver(),
        ]
        if data_aug:
            callbacks.append(ScheduledHyperParamSetter('learning_rate',
                                                       [(30, 1e-2), (60, 1e-3), (85, 1e-4), (95, 1e-5), (105, 1e-6)]))
        callbacks.append(HumanHyperParamSetter('learning_rate'))
        infs = [ClassificationError('wrong-top1', 'val-error-top1'),
                ClassificationError('wrong-top5', 'val-error-top5')]
        if nr_tower == 1:
            # single-GPU inference with queue prefetch
            callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
        else:
            # multi-GPU inference (with mandatory queue prefetch)
            callbacks.append(DataParallelInferenceRunner(
                dataset_val, infs, list(range(nr_tower))))

    return AutoResumeTrainConfig(
        model=model,
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=5000 if TOTAL_BATCH_SIZE == 256 else 10000,
        max_epoch=110 if data_aug else 64,
        nr_tower=nr_tower
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--fake', help='use fakedata to test or benchmark this model', action='store_true')
    parser.add_argument('--data_format', help='specify NCHW or NHWC',
                        type=str, default='NCHW')
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=18, choices=[18, 34, 50, 101, 152])
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--mode', choices=['resnet', 'preact', 'preact_typeA', 'vgg', 'alexnet', 'googlenet', 'densenet'],
                        help='variants of resnet to use. resnet, preact and preact_typeA are all the type of resnet. resnet means only quantizing weight; preact means pre-activation resnet with quantized both weight and activation and the shortcut type is type A; preact_typeA means pre-activation resnet with quantized both weight and activation and the shortcut type is type B.',
                        default='resnet')
    parser.add_argument('--logdir_id', help='identify of logdir',
                        type=str, default='')
    parser.add_argument('--qw', help='weight quant',
                        type=int, default=1)
    parser.add_argument('--qa', help='activation quant',
                        type=int, default=2)
    parser.add_argument('--wd', help='weight decay',
                        type=float, default=5e-5)
    parser.add_argument('--data_aug', help='data augmentation',
                        action='store_true')
    parser.add_argument('--batch_size', help='batch size',
                        type=int, default=256)
    parser.add_argument('--input_size', help='input size',
                        type=int, default=224)
    parser.add_argument('--lr', help='learning rate',
                        type=float, default=0.1)
    parser.add_argument('--log_path', help='path of log',
                        type=str, default='')
    parser.add_argument('--action', help='action type',
                        type=str, default='')
    args = parser.parse_args()

    TOTAL_BATCH_SIZE = args.batch_size
    imagenet_utils.DEFAULT_IMAGE_SHAPE = args.input_size

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = Model(args.depth, args.data_format, args.mode, args.wd, args.qw, args.qa, learning_rate=args.lr, data_aug=args.data_aug)
    if args.eval:
        batch = 100  # something that can run on one gpu
        ds = get_data('val', batch)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
    else:
        if args.log_path == '':
            logger.set_logger_dir(
                os.path.join('train_log', 'imagenet_resnet_d' + str(args.depth) + args.logdir_id), action=None if args.action == '' else args.action)
        else:
            logger.set_logger_dir(args.log_path + '/train_log/' + args.logdir_id, action=None if args.action == '' else args.action)

        config = get_config(model, fake=args.fake, data_aug=args.data_aug)
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = SyncMultiGPUTrainerReplicated(max(get_nr_gpu(), 1))
        launch_train_with_config(config, trainer)
