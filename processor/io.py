#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
import random
 
# torch
import torch
import torch.nn as nn

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class
import os.path as osp
import os
import glob
from processor.utils.parameters import get_basic_parser


def get_dir(root_dir, sub_dir=None):
    if sub_dir:
        dir = osp.join(root_dir, sub_dir)
    else:
        dir = root_dir
    if not osp.isdir(dir):
        os.makedirs(dir)
    return dir


class IO:
    """
        IO Processor
    """

    def __init__(self, argv=None):

        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        self.init_io()

    def load_arg(self, argv=None):
        parser = self.get_parser()

        # load arg form config file
        p = parser.parse_args(argv)
        if p.config is not None:
            # load config file
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f)

            # update parser from config file
            key = vars(p).keys()
            for k in default_arg.keys():
                if k not in key:
                    print('Unknown Arguments: {}'.format(k))
                    assert k in key

            parser.set_defaults(**default_arg)

        self.arg = parser.parse_args(argv)

    def init_environment(self):
        # gpu
        if self.arg.use_gpu:
            gpus = torchlight.visible_gpu(self.arg.device)
            torchlight.occupy_gpu(gpus)
            self.gpus = gpus
            self.dev = "cuda:0"
        else:
            self.dev = "cpu"

        # random seed
        if self.arg.fix_random:
            SEED = 0
            #torch.backends.cudnn.deterministic = True
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(SEED)
            random.seed(SEED)

        # dir
        self.work_dir = get_dir(self.arg.work_dir, None)
        self.config_dir = get_dir(self.arg.work_dir, 'config')
        self.checkpoint_dir = get_dir(self.arg.work_dir, 'checkpoints')
        self.log_dir = get_dir(self.arg.work_dir, 'log')
        self.tflog_dir = get_dir(self.arg.work_dir, 'tflog')
        self.test_dir = get_dir(self.arg.work_dir, 'test_results')

    def init_io(self):
        self.io = torchlight.IO(
            self.arg.work_dir,
            self.arg.session,
            save_log=self.arg.save_log,
            print_log=self.arg.print_log)
        if self.arg.phase == 'train':
            self.io.save_arg(self.arg)

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))

    def load_weights(self):
        if self.arg.weights:
            self.model = self.io.load_weights(self.model, self.arg.weights,
                                              self.arg.ignore_weights)

    def gpu(self):
        # move modules to gpu
        self.model = self.model.to(self.dev)
        for name, value in vars(self).items():
            cls_name = str(value.__class__)
            if cls_name.find('torch.nn.modules') != -1:
                setattr(self, name, value.to(self.dev))

    def parallel(self):
        # model parallel
        if self.arg.use_gpu and len(self.gpus) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpus)

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

    @staticmethod
    def get_parser(add_help=False):
        return get_basic_parser(add_help)
