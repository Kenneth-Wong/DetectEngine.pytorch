#!/usr/bin/env python
# pylint: disable=W0201
# This is the reimplemented version of Neural Motifs by Kenneth-Wong,
# please refer to https://github.com/rowanz/neural-motifs
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from net.utils.bbox_transform import bbox_transform_inv, clip_boxes_batch
from compiled_modules.nms.nms_wrapper import nms

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from ..processor import Processor
from ..io import get_dir
import os.path as osp
import glob
from net.utils.net_utils import clip_gradient
import time
import pickle


class frcnn_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        FasterRCNN = import_class(self.arg.model)
        self.model = FasterRCNN(self.classes, pretrained=True, class_agnostic=self.arg.mix_args['class_agnostic'],
                                **vars(self.arg))
        self.model.create_architecture()
        # self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        params = []
        self.lr = self.arg.opt_args['lr']
        for key, value in dict(self.model.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [
                        {'params': [value], 'lr': self.arg.opt_args['lr'] * (self.arg.opt_args['double_bias'] + 1),
                         'weight_decay': self.arg.opt_args['bias_decay'] and self.arg.opt_args['weight_decay'] or 0}]
                else:
                    params += [{'params': [value], 'lr': self.arg.opt_args['lr'],
                                'weight_decay': self.arg.opt_args['weight_decay']}]

        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(params, momentum=self.arg.opt_args['momentum'],
                                       nesterov=self.arg.opt_args['nesterov'])
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(params)
        else:
            raise ValueError()

    def resume_checkpoint(self):
        if self.arg.resume:
            if self.arg.checkpoint is not None:
                assert osp.isfile(self.arg.checkpoint), 'Cannot find checkpoint %s.' % (self.arg.checkpoint)
                checkpoint = self.arg.checkpoint
            else:
                # automatically find checkpoint
                files = osp.join(self.checkpoint_dir, self.arg.snapshot_prefix + '_{}_*.pth'.format(self.arg.session))
                files = glob.glob(files)
                if len(files) == 0:
                    raise ValueError('Cnnot find any checkpoints.')
                files.sort(key=osp.getmtime)
                checkpoint = files[-1]

            self.ckpt = checkpoint
            print("Loading checkpoint %s" % (checkpoint))
            cpt = torch.load(checkpoint)
            self.arg.session = cpt['session']
            self.arg.opt_args['start_epoch'] = cpt['epoch']
            self.model.load_state_dict(cpt['model'])
            self.optimizer.load_state_dict(cpt['optimizer'])
            self.arg.opt_args['lr'] = self.optimizer.param_groups[0]['lr']
            self.lr = self.arg.opt_args['lr']
            if 'pooling_mode' in cpt.keys():
                self.arg.mix_args['pooling_mode'] = cpt['pooling_mode']
            if 'class_agnostic' in cpt.keys():
                self.arg.mix_args['class_agnostic'] = cpt['class_agnostic']
            print("Loaded checkpoint %s" % checkpoint)

    def add_additional_save_state(self):
        return {'pooling_mode': self.arg.mix_args['pooling_mode'],
                'class_agnostic': self.arg.mix_args['class_agnostic']}

    def adjust_lr(self):
        epoch = self.meta_info['epoch']
        if epoch in self.arg.opt_args['step_epoch']:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.arg.opt_args['gamma']
            self.lr *= self.arg.opt_args['gamma']

    def initialize_data_holder(self):
        self.inputs = []
        #if self.arg.use_gpu and len(self.gpus) == 1:
        for i in range(4):
            self.inputs.append(Variable(torch.FloatTensor([1]).cuda()))
            # self.im_data = Variable(torch.FloatTensor(1).cuda())
            # self.im_info = Variable(torch.FloatTensor(1).cuda())
            # self.gt_boxes = Variable(torch.FloatTensor(1).cuda())
            # self.num_boxes = Variable(torch.LongTensor(1).cuda())

    def train(self):
        self.model.train()

        loader = self.data_loader['train']
        train_iters_per_epoch = self.train_size // self.arg.train_args['ims_per_batch']
        self.meta_info['iters_per_epoch'] = train_iters_per_epoch
        data_iter = iter(loader)
        loss_temp = 0
        epoch_start = time.time()
        iter_start = epoch_start
        eval_time = 0

        self.adjust_lr()

        for step in range(train_iters_per_epoch):
            self.meta_info['iter'] = self.meta_info['iter'] + step
            self.meta_info['step'] = step
            # get data
            data = next(data_iter)
            for i in range(4):
                self.inputs[i].data.resize_(data[i].size()).copy_(data[i])
            # self.im_data.data.resize_(data[0].size()).copy_(data[0])
            # self.im_info.data.resize_(data[1].size()).copy_(data[1])
            # self.gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            # self.num_boxes.data.resize_(data[3].size()).copy_(data[3])

            # forward
            self.model.zero_grad()
            rois, cls_prob, bbox_pred, rpn_label, \
            rpn_cls_loss, rpn_box_loss, RCNN_cls_loss, RCNN_box_loss, rois_label = self.model(*(self.inputs))
            loss = rpn_cls_loss.mean() + rpn_box_loss.mean() + RCNN_cls_loss.mean() + RCNN_box_loss.mean()
            loss_temp += loss.item()

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            if 'vgg16' in self.arg.model:
                clip_gradient(self.model, 10)
            self.optimizer.step()

            if self.meta_info['step'] % self.arg.log_interval == 0:
                iter_end = time.time()
                eval_start = time.time()
                if step > 0:
                    loss_temp /= self.arg.log_interval

                loss_rpn_cls = rpn_cls_loss.mean().item()
                loss_rpn_box = rpn_box_loss.mean().item()
                loss_rcnn_cls = RCNN_cls_loss.mean().item()
                loss_rcnn_box = RCNN_box_loss.mean().item()
                rpn_fg_cnt = torch.sum(rpn_label.data.eq(1)).item()
                rpn_bg_cnt = torch.sum(rpn_label.data.eq(0)).item()
                fg_cnt = torch.sum(rois_label.data.ne(0)).item()
                bg_cnt = rois_label.data.numel() - fg_cnt

                self.iter_info['loss_avg'] = loss_temp
                self.iter_info['lr'] = self.lr
                self.iter_info['loss_rpn_cls'] = loss_rpn_cls
                self.iter_info['loss_rpn_box'] = loss_rpn_box
                self.iter_info['loss_rcnn_cls'] = loss_rcnn_cls
                self.iter_info['loss_rcnn_box'] = loss_rcnn_box
                self.iter_info['rpn_fg_cnt'] = rpn_fg_cnt
                self.iter_info['rpn_bg_cnt'] = rpn_bg_cnt
                self.iter_info['fg_cnt'] = fg_cnt
                self.iter_info['bg_cnt'] = bg_cnt
                self.iter_info['time_cost'] = (iter_end - iter_start) / self.arg.log_interval if step > 0 else (
                    iter_end - iter_start)

                self.show_iter_info()
                eval_end = time.time()
                eval_time += (eval_end - eval_start)
                loss_temp = 0
                iter_start = time.time()
        epoch_end = time.time()

        self.epoch_info['time_cost'].append((epoch_end - epoch_start - eval_time) / 3600)
        self.show_epoch_info()
        # self.io.print_timer()

    def test(self, evaluation=True):
        self.model.eval()
        start = time.time()
        max_per_image = 100
        thresh = 0.0
        empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
        all_boxes = [[[] for _ in range(self.test_size)] for _ in range(len(self.classes))]
        loader = self.data_loader['test']
        data_iter = iter(loader)
        output_dir = get_dir(self.test_dir, self.ckpt.split('/')[-1].replace('.pth', '_')+self.imdb_test.name)
        _t = {'im_detect': time.time(), 'misc': time.time()}

        for i in range(self.test_size):
        #for i in range(10):
            data = next(data_iter)
            for k in range(4):
                self.inputs[k].data.resize_(data[k].size()).copy_(data[k])
            det_tic = time.time()
            rois, cls_prob, bbox_pred, rpn_label, \
            rpn_cls_loss, rpn_box_loss, RCNN_cls_loss, RCNN_box_loss, rois_label = self.model(*(self.inputs))
            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            if self.arg.test_args['bbox_reg']:
                box_deltas = bbox_pred.data
                if self.arg.train_args['bbox_normalize_targets_precomputed']:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                        self.arg.train_args['bbox_normalize_stds']).cuda() + torch.FloatTensor(
                        self.arg.train_args['bbox_normalize_means']).cuda()
                    box_deltas = box_deltas.view(1, -1, 4) if self.arg.mix_args['class_agnostic'] else box_deltas.view(
                        1, -1, 4 * len(self.classes))
                pred_boxes = bbox_transform_inv(boxes, box_deltas)
                pred_boxes = clip_boxes_batch(pred_boxes, data[1].cuda(), 1)
            else:
                pred_boxes = boxes if self.arg.mix_args['class_agnostic'] else np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= data[1][0][2].item()

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            det_time = det_toc - det_tic
            misc_tic = time.time()

            for j in range(1, len(self.classes)):
                inds = torch.nonzero(scores[:, j]>thresh).view(-1)
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if self.arg.mix_args['class_agnostic']:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds, j*4:(j+1)*4]
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, self.arg.test_args['nms'], force_cpu=not self.arg.mix_args['use_gpu_nms'])
                    cls_dets = cls_dets[keep.view(-1).long()]
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array

            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]] for j in range(1, len(self.classes)))
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, len(self.classes)):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]
            misc_toc = time.time()
            nms_time = misc_toc - misc_tic
            #print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
            #                 .format(i + 1, self.test_size, det_time, nms_time))
            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                             .format(i + 1, self.test_size, det_time, nms_time))
            sys.stdout.flush()

        with open(osp.join(output_dir, 'detections.pkl'), 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        self.imdb_test.evaluate_detections(all_boxes, output_dir)

        end = time.time()
        print("test time: %0.4fs" % (end - start))

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='faster RCNN')

        # region arguments yapf: disable
        # evaluation
        # endregion yapf: enable

        return parser
