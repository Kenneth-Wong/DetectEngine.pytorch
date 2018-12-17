import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np

from compiled_modules.roi_align.modules.roi_align import RoIAlign, RoIAlignAvg
from compiled_modules.roi_pooling.modules.roi_pool import _RoIPooling
from compiled_modules.roi_crop.modules.roi_crop import _RoICrop
from net.utils.proposal_target_layer_cascade import _ProposalTargetLayer
from net.utils.rpn import _RPN
from net.utils.net_utils import _smooth_l1_loss

import time


class fasterRCNN(nn.Module):
    """faster RCNN"""
    def __init__(self, classes, class_agnostic, **args):
        super(fasterRCNN, self).__init__()
        self.classes = classes
        self.num_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.args = args

        # some params of backbone network
        # self.dout_base_model, self.RCNN_base, self.RCNN_top, self._head_to_tail,
        # self.RCNN_bbox_pred, self.RCNN_cls_score, _init_modules

        # rpn
        self.RCNN_rpn = _RPN(self.dout_base_model, **args)

        # proposal target, you can change it with another sampling way
        self.RCNN_proposal_target = _ProposalTargetLayer(self.num_classes, **args)

        # pooling methods
        self.RCNN_roi_pool = _RoIPooling(args['mix_args']['pooling_size'], args['mix_args']['pooling_size'],
                                         1.0 / args['mix_args']['feat_stride'][0])
        self.RCNN_roi_align = RoIAlignAvg(args['mix_args']['pooling_size'], args['mix_args']['pooling_size'],
                                         1.0 / args['mix_args']['feat_stride'][0])

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        base_feat = self.RCNN_base(im_data)  # b*512*H*W

        # rpn
        rois, rpn_cls_loss, rpn_box_loss, rpn_label = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # proposal target if training
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_weights, rois_outside_weights = roi_data

            rois_label = Variable(rois_label.view(-1).long())  ## 256b
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))  ## 256b*4
            rois_inside_weights = Variable(rois_inside_weights.view(-1, rois_inside_weights.size(2)))  ## 256b*4
            rois_outside_weights = Variable(rois_outside_weights.view(-1, rois_outside_weights.size(2)))

        else:
            rois_label = None
            rois_target = None
            rois_inside_weights = None
            rois_outside_weights = None
            rpn_cls_loss = 0
            rpn_box_loss = 0
            rpn_label = None

        rois = Variable(rois)  ## b*256*5

        ## pooling
        if self.args['mix_args']['pooling_mode'] == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
        elif self.args['mix_args']['pooling_mode'] == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))

        pooled_feat = self._head_to_tail(pooled_feat)

        ## RCNN loss
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)  ## 256b*(4C) or 256b*4(class_agnostic)
        if self.training and not self.class_agnostic:
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1)/4), 4)  # 256b*C*4
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)  ## 256b*4

        cls_score = self.RCNN_cls_score(pooled_feat)  # 256b*C
        cls_prob = F.softmax(cls_score, 1)

        RCNN_cls_loss = 0
        RCNN_box_loss = 0

        if self.training:
            RCNN_cls_loss = F.cross_entropy(cls_score, rois_label)
            RCNN_box_loss = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_weights, rois_outside_weights)
            # fix torch0.4 bug for mGPU
            RCNN_cls_loss = torch.unsqueeze(RCNN_cls_loss, 0)
            RCNN_box_loss = torch.unsqueeze(RCNN_box_loss, 0)
            rpn_cls_loss = torch.unsqueeze(rpn_cls_loss, 0)
            rpn_box_loss = torch.unsqueeze(rpn_box_loss, 0)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_label, \
               rpn_cls_loss, rpn_box_loss, RCNN_cls_loss, RCNN_box_loss, rois_label

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
                m.bias.data.zero_()
        train_args = self.args['train_args']
        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, train_args['truncated'])
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, train_args['truncated'])
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, train_args['truncated'])
        normal_init(self.RCNN_cls_score, 0, 0.01, train_args['truncated'])
        normal_init(self.RCNN_bbox_pred, 0, 0.001, train_args['truncated'])

    def create_architecture(self):
        self._init_modules()
        self._init_weights()



