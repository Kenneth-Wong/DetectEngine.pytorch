import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np

from net.utils.net_utils import Conv2d, Conv2d_BatchNorm
from net.utils.yolo_target_layer import _YOLOTargetLayer
from net.utils.yolo_preprocess_layer import _YOLOPreprocessLayer
from compiled_modules.reorg.reorg_layer import ReorgLayer

import time


def _make_layers(in_channels, net_cfg):
    layers = []

    if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
        for sub_cfg in net_cfg:
            layer, in_channels = _make_layers(in_channels, sub_cfg)
            layers.append(layer)
    else:
        for item in net_cfg:
            if item == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels, ksize = item
                layers.append(Conv2d_BatchNorm(in_channels, out_channels, ksize, same_padding=True))
                in_channels = out_channels

    return nn.Sequential(*layers), in_channels


class yolov2(nn.Module):
    """yolov2"""

    def __init__(self, classes, **args):
        super(yolov2, self).__init__()
        self.classes = classes  ## yolo does not need __background__
        self.num_classes = len(classes)
        self.args = args
        self.num_anchors = len(self.args['mix_args']['anchors'])

        # some params of backbone network
        # self.dout_base_model, self.yolov2_base, _init_modules

        # head
        self.conv3, c3 = _make_layers(self.dout_base_model, [(1024, 3), (1024, 3)])
        stride = 2
        in_c4 = (self.dout_middle_model * stride * stride + c3) if self.dout_middle_model else c3
        self.conv4, c4 = _make_layers(in_c4, [(1024, 3)])
        out_channels = self.num_anchors * (self.num_classes + 5)
        self.conv_target = Conv2d(self.dout_base_model, out_channels, 1, 1, relu=False)

        # resize input layer
        self.yolo_preprocess = _YOLOPreprocessLayer(**args)

        # reorg layer
        self.reorg = ReorgLayer(stride=2)

        # target layer
        self.yolo_target = _YOLOTargetLayer(self.num_classes, **args)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, size_index=0):
        batch_size = im_data.size(0)
        im_data, gt_boxes = self.yolo_preprocess(im_data, gt_boxes, size_index)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        if isinstance(self.yolov2_base, nn.Sequential):
            base_feat = self.yolov2_base(im_data)  # b*1024*13*13
        else:
            ## residual connection, self.conv1s and self.conv2 are defined inside darknet19
            conv1s_res = self.conv1s(im_data)
            conv2_res = self.conv2(conv1s_res)
            conv3_res = self.conv3(conv2_res)
            conv1s_reorg = self.reorg(conv1s_res)
            cat_1_3 = torch.cat([conv1s_reorg, conv3_res], 1)
            base_feat = self.conv4(cat_1_3)

        predictions = self.conv_target(base_feat)  # b*(5*25)*13*13
        predictions = predictions.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_anchors,
                                                                        self.num_classes + 5)

        # tx, ty, tw, th, to -> sig(tx), sig(ty), exp(tw), exp(th), sig(to)
        xy_pred = F.sigmoid(predictions[:, :, :, 0:2])
        wh_pred = torch.exp(predictions[:, :, :, 2:4])
        bbox_pred = torch.cat([xy_pred, wh_pred], 3)  # b*169*5*4
        iou_pred = F.sigmoid(predictions[:, :, :, 4:5])  # b*169*5*1 (Note: 4:5 keeps the dimension sized 1 while 4 not)
        score_pred = predictions[:, :, :, 5:].contiguous()  # b*169*5*20
        prob_pred = F.softmax(score_pred.view(-1, score_pred.size()[-1])).view_as(score_pred)

        YOLO_cls_loss = 0.
        YOLO_box_loss = 0.
        YOLO_iou_loss = 0.

        if self.training:
            _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask = self.yolo_target(bbox_pred, iou_pred,
                                                                                          gt_boxes,
                                                                                          num_boxes, size_index)
            num_all_boxes = torch.sum(num_boxes)
            YOLO_cls_loss = F.mse_loss(prob_pred * _class_mask, _classes * _class_mask,
                                       size_average=False) / num_all_boxes

            YOLO_box_loss = F.mse_loss(bbox_pred * _box_mask, _boxes * _box_mask, size_average=False) / num_all_boxes
            YOLO_iou_loss = F.mse_loss(iou_pred * _iou_mask, _ious * _iou_mask, size_average=False) / num_all_boxes

            YOLO_box_loss = torch.unsqueeze(YOLO_box_loss, 0)
            YOLO_iou_loss = torch.unsqueeze(YOLO_iou_loss, 0)
            YOLO_cls_loss = torch.unsqueeze(YOLO_cls_loss, 0)
        return bbox_pred, iou_pred, prob_pred, YOLO_cls_loss, YOLO_box_loss, YOLO_iou_loss

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                # m.bias.data.zero_() : no bias here for conv

        def constant_init(m):
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)

        # train_args = self.args['train_args']
        # for m in self.conv3:  # m is a conv2d-BN layer
        #     normal_init(m.conv, 0, 0.01, train_args['truncated'])
        #     constant_init(m.bn)
        # for m in self.conv4:
        #     normal_init(m.conv, 0, 0.01, train_args['truncated'])
        #     constant_init(m.bn)
        # normal_init(self.conv_target.conv, 0, 0.01, train_args['truncated'])

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
