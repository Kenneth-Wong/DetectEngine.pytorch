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
from net.utils.bbox_transform import bbox_transform_inv, clip_boxes_batch, yolo_to_bbox
from compiled_modules.nms.nms_wrapper import nms

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from ..processor import Processor
from ..io import get_dir
from feeder.utils.roidb import combined_roidb
from feeder.utils.minibatch import get_size_minibatch
import os.path as osp
import glob
from net.utils.net_utils import clip_gradient
import time
import pickle


class yolov2_Processor_v2(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        yolo = import_class(self.arg.model)
        self.model = yolo(self.classes, pretrained=True, **vars(self.arg))
        self.model.create_architecture()

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

    def adjust_lr(self):
        epoch = self.meta_info['epoch']
        if epoch in self.arg.opt_args['step_epoch']:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.arg.opt_args['gamma']
            self.lr *= self.arg.opt_args['gamma']

    def initialize_data_holder(self):
        self.inputs = []
        for i in range(4):
            self.inputs.append(Variable(torch.FloatTensor([1]).cuda()))
            # self.im_data = Variable(torch.FloatTensor(1).cuda())
            # self.im_info = Variable(torch.FloatTensor(1).cuda())
            # self.gt_boxes = Variable(torch.FloatTensor(1).cuda())
            # self.num_boxes = Variable(torch.LongTensor(1).cuda())

    def show_iter_info(self):
        info = '\t[session %d][epoch %2d][iter %4d/%4d][time/iter %.4f s][lr %.2e]' % \
               (self.arg.session, self.meta_info['epoch'], self.meta_info['step'], self.meta_info['iters_per_epoch'],
                self.iter_info['time_cost'], self.iter_info['lr'])

        for k, v in self.iter_info.items():
            if k.startswith('loss'):
                info += '\t{}: {:.4f}'.format(k, v)

        self.io.print_log(info)

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
            self.roidb = roidb
            train_size = len(roidb)
            self.train_size = train_size
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(roidb, num_classes, need_bg=self.need_bg, **vars(self.arg)),
                batch_size=self.arg.train_args['ims_per_batch'],
                num_workers=self.arg.num_worker * torchlight.ngpu(self.arg.device),
                drop_last=True)

            if imdbval_name:
                imdb_val, roidb_val, ratio_list_val, ratio_index_val = combined_roidb(imdbval_name, training=False,
                                                                                      **vars(self.arg))
                self.need_val = True
                self.val_size = len(roidb_val)
                self.imdb_val = imdb_val
                self.roidb_val = roidb_val
                self.data_loader['val'] = torch.utils.data.DataLoader(
                    dataset=Feeder(roidb_val, num_classes, need_bg=self.need_bg, training=False, **vars(self.arg)),
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
            self.roidb_test = roidb_test
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(roidb_test, num_classes, need_bg=self.need_bg, training=False, **vars(self.arg)),
                batch_size=self.arg.test_args['ims_per_batch'],
                shuffle=False,
                num_workers=0, pin_memory=True)

    def train(self):
        self.model.train()

        loader = self.data_loader['train']
        loader.dataset.reset_size_batch()
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

            data = next(data_iter)

            for i in range(4):
                self.inputs[i].data.resize_(data[i].size()).copy_(data[i])
            # self.im_data.data.resize_(data[0].size()).copy_(data[0])
            # self.im_info.data.resize_(data[1].size()).copy_(data[1])
            # self.gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            # self.num_boxes.data.resize_(data[3].size()).copy_(data[3])

            # forward
            self.model.zero_grad()
            bbox_pred, iou_pred, prob_pred, YOLO_cls_loss, YOLO_box_loss, YOLO_iou_loss = self.model(
                *(self.inputs))
            loss = YOLO_box_loss.mean() + YOLO_iou_loss.mean() + YOLO_cls_loss.mean()
            loss_temp += loss.item()

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.meta_info['step'] % self.arg.log_interval == 0:
                iter_end = time.time()
                eval_start = time.time()
                if step > 0:
                    loss_temp /= self.arg.log_interval

                loss_yolo_box = YOLO_box_loss.mean().item()
                loss_yolo_iou = YOLO_iou_loss.mean().item()
                loss_yolo_cls = YOLO_cls_loss.mean().item()

                self.iter_info['loss_avg'] = loss_temp
                self.iter_info['lr'] = self.lr
                self.iter_info['loss_yolo_box'] = loss_yolo_box
                self.iter_info['loss_yolo_iou'] = loss_yolo_iou
                self.iter_info['loss_yolo_cls'] = loss_yolo_cls
                self.iter_info['time_cost'] = (iter_end - iter_start) / self.arg.log_interval if step > 0 else (
                    iter_end - iter_start)

                self.show_iter_info()
                eval_end = time.time()
                eval_time += (eval_end - eval_start)
                loss_temp = 0
                iter_start = time.time()
            #if self.meta_info['step'] > 0 and self.meta_info['step'] % self.arg.train_args['batch_per_size'] == 0:
            #    size_index = np.random.randint(0, len(self.arg.mix_args['input_size']))
            #    print("change image size to {}*{}".format(self.arg.mix_args['input_size'][size_index][0],
            #                                              self.arg.mix_args['input_size'][size_index][1]))
        epoch_end = time.time()

        self.epoch_info['time_cost'].append((epoch_end - epoch_start - eval_time) / 3600)
        self.show_epoch_info()
        # self.io.print_timer()

    def test(self, evaluation=True):
        self.model.eval()
        start = time.time()
        max_per_image = 300
        thresh = 0.001
        empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
        all_boxes = [[[] for _ in range(self.test_size)] for _ in range(len(self.classes) + 1)]  ## write file with bg
        loader = self.data_loader['test']
        size_index = self.arg.test_feeder_args['size_index']
        loader.dataset.reset_size_batch(size_index)
        data_iter = iter(loader)
        input_size = self.arg.mix_args['input_size']
        output_size = self.arg.mix_args['output_size']
        w_in, h_in = input_size[size_index]
        w_out, h_out = output_size[size_index]
        output_dir = get_dir(self.test_dir,
                             self.ckpt.split('/')[-1].replace('.pth', '_') + self.imdb_test.name + '/size_{}'.format(
                                 w_in))
        for i in range(self.test_size):
            # for i in range(100):
            data = next(data_iter)

            for k in range(4):
                self.inputs[k].data.resize_(data[k].size()).copy_(data[k])
            det_tic = time.time()
            bbox_pred, iou_pred, prob_pred, YOLO_cls_loss, YOLO_box_loss, YOLO_iou_loss = self.model(
                *(self.inputs))

            anchors = torch.from_numpy(np.array(self.arg.mix_args['anchors'])).float().cuda()
            bbox_pred = yolo_to_bbox(bbox_pred, anchors, h_out, w_out)
            bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)
            bbox_pred[:, :, 0::2] = bbox_pred[:, :, 0::2] * data[1][0][1].item()
            bbox_pred[:, :, 1::2] = bbox_pred[:, :, 1::2] * data[1][0][0].item()

            bbox_pred = bbox_pred.squeeze()
            iou_pred = iou_pred.view(-1)
            prob_pred = prob_pred.view(-1, prob_pred.size(-1))
            cls_inds = torch.argmax(prob_pred, dim=1)
            prob_pred = prob_pred[torch.arange(prob_pred.size(0)).long(), cls_inds]
            scores = iou_pred * prob_pred

            keep = torch.nonzero(scores >= thresh).view(-1)
            bbox_pred = bbox_pred[keep]
            scores = scores[keep]
            cls_inds = cls_inds[keep]

            det_toc = time.time()
            det_time = det_toc - det_tic
            misc_tic = time.time()

            # NMS
            keep = torch.zeros(bbox_pred.size(0)).long().cuda()
            for j in range(len(self.classes)):
                inds = torch.nonzero(cls_inds == j).view(-1)
                if inds.numel() > 0:
                    c_bboxes = bbox_pred[inds]
                    c_scores = scores[inds]
                    _, order = torch.sort(c_scores, 0, True)
                    cls_dets = torch.cat((c_bboxes, c_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]
                    order_inds = inds[order]
                    c_keep = nms(cls_dets, self.arg.test_args['nms'], force_cpu=not self.arg.mix_args['use_gpu_nms'])
                    keep[order_inds[c_keep.view(-1).long()]] = 1

            keep = torch.nonzero(keep > 0).view(-1).long()
            bbox_pred = bbox_pred[keep]
            scores = scores[keep]
            cls_inds = cls_inds[keep]
            bbox_pred = clip_boxes_batch(bbox_pred.unsqueeze(0), data[1].cuda(), 1).squeeze()

            for j in range(len(self.classes)):
                inds = torch.nonzero(cls_inds == j).view(-1)
                if inds.numel() > 0:
                    c_bboxes = bbox_pred[inds]
                    c_scores = scores[inds]
                    cls_dets = torch.cat((c_bboxes, c_scores.unsqueeze(1)), 1)
                    all_boxes[j + 1][i] = cls_dets.data.cpu().numpy()
                    # cls_dets = cls_dets[c_keep.view(-1).long()]
                    # all_boxes[j+1][i] = cls_dets.data.cpu().numpy()
                else:
                    all_boxes[j + 1][i] = empty_array

            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]] for j in range(1, len(self.classes) + 1))
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(0, len(self.classes)):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]
            misc_toc = time.time()
            nms_time = misc_toc - misc_tic
            # print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
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
            description='yolov2')

        # region arguments yapf: disable
        # evaluation
        # endregion yapf: enable

        return parser
