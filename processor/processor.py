#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class
from .io import IO
from feeder.utils.roidb import combined_roidb
import os.path as osp
from processor.utils.parameters import get_detbase_parser


class Processor(IO):
    """
        Base Processor
    """

    def __init__(self, argv=None):

        self.load_arg(argv)
        self.init_environment()
        self.load_data()
        self.load_model()
        self.load_optimizer()
        self.gpu()  # must move to gpu first
        self.resume_checkpoint()
        self.parallel()  # then parallel, in order to make the checkpoint loaded successfully without 'module.'
        self.init_io()

    def init_environment(self):

        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0, step=0)

    def load_optimizer(self):
        pass

    def resume_checkpoint(self):
        pass

    def load_data(self):
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
        self.need_val = False
        self.need_bg = not ('yolo' in self.arg.model)  ## yolo: False
        imdb_name = self.arg.train_dataset
        imdbval_name = self.arg.val_dataset
        imdbtest_name = self.arg.test_dataset
        self.train_size = 0
        self.val_size = 0
        self.test_size = 0
        self.imdb, self.imdb_val, self.imdb_test = None, None, None

        Feeder = import_class(self.arg.feeder)
        sampler = import_class(self.arg.sampler) if self.arg.sampler else None
        self.data_loader = dict()
        self.classes = None

        if self.arg.phase == 'train':
            assert imdb_name is not None, print('Training data is not provoided.')
            imdb, roidb, ratio_list, ratio_index = combined_roidb(imdb_name, **vars(self.arg))
            self.classes = imdb.classes if self.need_bg else imdb.classes[1:]
            num_classes = imdb.num_classes if self.need_bg else imdb.num_classes - 1
            self.imdb = imdb
            train_size = len(roidb)
            self.train_size = train_size
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(roidb, ratio_list, ratio_index, num_classes, need_bg=self.need_bg, **vars(self.arg)),
                batch_size=self.arg.train_args['ims_per_batch'],
                sampler=sampler(train_size, self.arg.train_args['ims_per_batch']),
                num_workers=self.arg.num_worker * torchlight.ngpu(self.arg.device),
                drop_last=True)

            if imdbval_name:
                imdb_val, roidb_val, ratio_list_val, ratio_index_val = combined_roidb(imdbval_name, training=False,
                                                                                      **vars(self.arg))
                self.need_val = True
                self.val_size = len(roidb_val)
                self.imdb_val = imdb_val
                self.data_loader['val'] = torch.utils.data.DataLoader(
                    dataset=Feeder(roidb_val, ratio_list_val, ratio_index_val, num_classes, training=False,
                                   need_bg=self.need_bg, **vars(self.arg)),
                    batch_size=self.arg.test_args['ims_per_batch'],
                    shuffle=False,
                    num_workers=self.arg.num_worker * torchlight.ngpu(self.arg.device))
        else:
            assert imdbtest_name is not None, print('Test data is not provoided.')
            imdb_test, roidb_test, ratio_list_test, ratio_index_test = combined_roidb(imdbtest_name, training=False,
                                                                                      **vars(self.arg))
            self.classes = imdb_test.classes if self.need_bg else imdb_test.classes[1:]
            num_classes = imdb_test.num_classes if self.need_bg else imdb_test.num_classes - 1
            self.test_size = len(roidb_test)
            self.imdb_test = imdb_test
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(roidb_test, ratio_list_test, ratio_index_test, num_classes, training=False,
                               need_bg=self.need_bg, **vars(self.arg)),
                batch_size=self.arg.test_args['ims_per_batch'],
                shuffle=False,
                num_workers=0, pin_memory=True)

    def show_epoch_info(self):
        info = '\t**EPOCH DONE**\t[session %d][epoch %2d]\t time/epoch %.4f h' % (
            self.arg.session, self.meta_info['epoch'], np.mean(self.epoch_info['time_cost']))
        self.io.print_log(info)
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):
        info = '\t[session %d][epoch %2d][iter %4d/%4d][time/iter %.4f s][lr %.2e]' % \
               (self.arg.session, self.meta_info['epoch'], self.meta_info['step'], self.meta_info['iters_per_epoch'],
                self.iter_info['time_cost'], self.iter_info['lr'])
        info += '\t[rpn]fg/bg=(%4d/%4d)\t[rcnn]fg/bg=(%4d/%4d)' % (
            self.iter_info['rpn_fg_cnt'], self.iter_info['rpn_bg_cnt'], self.iter_info['fg_cnt'],
            self.iter_info['bg_cnt'])

        for k, v in self.iter_info.items():
            if k.startswith('loss'):
                info += '\t{}: {:.4f}'.format(k, v)

        self.io.print_log(info)

        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.iter_info)

    def train(self):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['mean loss'] = 0
        self.show_epoch_info()

    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.epoch_info['mean loss'] = 1
        self.show_epoch_info()

    def val(self):
        pass

    def add_additional_save_state(self):
        return {}

    def initialize_data_holder(self):
        pass

    def start(self):
        # prepare tfboard
        if self.arg.use_tfboard:
            from tensorboardX import SummaryWriter
            logger = SummaryWriter(self.tflog_dir)

        # initialize input data holder
        self.initialize_data_holder()

        # training phase
        if self.arg.phase == 'train':
            self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

            start_epoch = self.arg.opt_args['start_epoch']
            max_epoch = self.arg.opt_args['max_epoch']
            assert (start_epoch <= max_epoch)
            self.epoch_info['time_cost'] = []
            self.meta_info['epoch'] = start_epoch
            self.meta_info['iter'] = start_epoch * (self.train_size // self.arg.train_args['ims_per_batch'])

            for epoch in range(start_epoch, max_epoch + 1):
                self.meta_info['epoch'] = epoch
                # training
                self.io.print_log('Training epoch: {}'.format(epoch))
                self.train()
                self.io.print_log('Done.')

                # save model
                if ((epoch - start_epoch + 1) % self.arg.save_interval == 0) or (
                            epoch == max_epoch):
                    filename = osp.join(self.checkpoint_dir,
                                        self.arg.snapshot_prefix + '_{}_{}.pth'.format(self.arg.session,
                                                                                       epoch))
                    save_state = {'session': self.arg.session, 'epoch': epoch + 1,
                                  'model': self.model.module.state_dict() if self.arg.use_gpu and len(
                                      self.gpus) > 1 else self.model.state_dict(),
                                  'optimizer': self.optimizer.state_dict()}
                    save_state.update(self.add_additional_save_state())
                    self.io.save_model(save_state, filename)

                # evaluation
                if self.need_val and (((epoch - start_epoch + 1) % self.arg.eval_interval == 0) or (
                            epoch == max_epoch)):
                    self.io.print_log('Eval epoch: {}'.format(epoch))
                    self.val()
                    self.io.print_log('Done.')
        # test phase
        elif self.arg.phase == 'test':
            # the path of checkpoint must be appointed
            assert self.arg.resume, 'Please set resume to True.'
            self.test()

    @staticmethod
    def get_parser(add_help=False):
        return get_detbase_parser(add_help)
