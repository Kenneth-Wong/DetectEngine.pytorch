import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .anchor_target_layer import _AnchorTargetLayer
from .proposal_layer import _ProposalLayer
from net.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time


class _RPN(nn.Module):
    """region proposal network"""

    def __init__(self, din, **args):
        super(_RPN, self).__init__()
        self.args = args
        self.din = din
        self.anchor_scales = args['mix_args']['anchor_scales']
        self.anchor_ratios = args['mix_args']['anchor_ratios']
        self.feat_stride = args['mix_args']['feat_stride'][0]

        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # rpn classification and regression layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2  # (bg/fg) * num_anchors
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios, **args)

        # anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios, **args)

        self.rpn_cls_loss = 0
        self.rpn_box_loss = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(input_shape[0], d, int(float(input_shape[1] * input_shape[2]) / float(d)), input_shape[3])
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):
        """
        :param base_feat: [b, 512, H, W]
        :param im_info: [b, 3]
        :param gt_boxes: [b, K, 5]
        :param num_boxes: [b,]
        :return:
        """
        batch_size = base_feat.size(0)
        rpn_conv = F.relu(self.RPN_Conv(base_feat), inplace=True)

        rpn_cls_score = self.RPN_cls_score(rpn_conv)  # b*18*H*W
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)  # b*2*9H*W
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, rpn_cls_score.size(1))

        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv)

        ## both necessary for train and test phrase
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data, im_info))

        self.rpn_cls_loss = 0
        self.rpn_box_loss = 0
        self.rpn_label = None

        if self.training:
            assert gt_boxes is not None
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)  # b*9HW*2
            rpn_label = rpn_data[0].view(batch_size, -1)  # b*9H*W -> b*9HW
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep.data)  # 256b*2
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_cls_loss = F.cross_entropy(rpn_cls_score, rpn_label)

            self.rpn_label = rpn_label

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_box_loss = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=3, dim=[1, 2, 3])

        return rois, self.rpn_cls_loss, self.rpn_box_loss, self.rpn_label
