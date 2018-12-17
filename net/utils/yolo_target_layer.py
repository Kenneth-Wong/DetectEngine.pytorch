import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch, yolo_to_bbox, anchor_intersections
import pdb
import math


class _YOLOTargetLayer(nn.Module):
    """
    Compute the targets for bounding box, iou, and classes predicted by YOLO.
    """

    def __init__(self, num_classes, **args):
        super(_YOLOTargetLayer, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self._input_size = args['mix_args']['input_size']
        self._output_size = args['mix_args']['output_size']
        self._feat_stride = args['mix_args']['feat_stride'][0]
        self._anchors = torch.from_numpy(np.array(args['mix_args']['anchors'], dtype=np.float))

    def forward(self, bbox_pred, iou_pred, gt_boxes, num_boxes, size_index):
        """
        build targets for bbox, iou, and classes
        :param bbox_pred: b*hw*5*4
        :param iou_pred:  b*hw*5*1
        :param gt_boxes:  b*N*5
        :param num_boxes: b
        :param size_index: int
        :return:
        """
        Win, Hin = self._input_size[size_index]
        Wout, Hout = self._output_size[size_index]
        batch_size, hw, num_anchors = bbox_pred.size(0), bbox_pred.size(1), bbox_pred.size(2)
        max_gt_num = gt_boxes.size(1)
        assert hw == Wout * Hout, 'Size mis match: the output feature map has size {} but it should be {}'.format(
            hw, Wout * Hout)
        self._anchors = self._anchors.type_as(gt_boxes)

        ## 1. Define the groundtruth info and the mask
        _classes = gt_boxes.new(batch_size, hw, num_anchors, self.num_classes).zero_()
        _class_mask = gt_boxes.new(batch_size, hw, num_anchors, 1).zero_()
        _ious = iou_pred.new(batch_size, hw, num_anchors, 1).zero_()
        _iou_mask = iou_pred.new(batch_size, hw, num_anchors, 1).zero_()
        _boxes = bbox_pred.new(batch_size, hw, num_anchors, 4).zero_()
        _boxes[:, :, :, 0:2] = 0.5
        _boxes[:, :, :, 2:4] = 1.0
        _box_mask = bbox_pred.new(batch_size, hw, num_anchors, 1).fill_(0.01)

        ## 2. With the bbox_pred(bias) and anchors, restore the real predicted boxes
        real_bbox_pred = yolo_to_bbox(bbox_pred, self._anchors, Hout, Wout)
        real_bbox_pred[:, :, :, 0::2] *= float(Win)
        real_bbox_pred[:, :, :, 1::2] *= float(Hin)  # b*hw*5*4

        ## 3. compute iou between predicted boxes and gt boxes to penalize the anchors which do not contain objects
        ious = bbox_overlaps_batch(real_bbox_pred.view(batch_size, -1, 4), gt_boxes)  # b*(hw*5)*N
        ious_reshaped = ious.view(batch_size, hw, num_anchors, -1)
        best_ious, _ = torch.max(ious, 2)
        best_ious = best_ious.view(batch_size, hw, num_anchors, 1).contiguous()
        iou_penalty = 0 - iou_pred[best_ious < self.args['train_args']['iou_thresh']]
        no_obj_coefficient = self.args['train_args']['noobject_scale'] * iou_penalty  # or noobject_scale directly
        _iou_mask[best_ious < self.args['train_args']['iou_thresh']] = no_obj_coefficient

        ## 4. locate the cell and compute the regression target for each gt boxes
        cell_w = self._feat_stride
        cell_h = self._feat_stride
        cx = (gt_boxes[:, :, 0] + gt_boxes[:, :, 2]) * 0.5 / cell_w  # b*N
        cy = (gt_boxes[:, :, 1] + gt_boxes[:, :, 3]) * 0.5 / cell_h
        cell_inds = (torch.floor(cy) * Wout + torch.floor(cx)).long()  # b*N

        target_boxes = gt_boxes.new(batch_size, max_gt_num, 4).zero_()
        target_boxes[:, :, 0] = cx - torch.floor(cx)  # bias, the regression target of predicted boxes
        target_boxes[:, :, 1] = cy - torch.floor(cy)
        target_boxes[:, :, 2] = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1) / self._feat_stride  # tw
        target_boxes[:, :, 3] = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1) / self._feat_stride

        ## 5. for each gt_boxes, match the best anchor
        gt_boxes_resize = gt_boxes[:, :, :4].clone()
        gt_boxes_resize[:, :, 0::2] /= self._feat_stride
        gt_boxes_resize[:, :, 1::2] /= self._feat_stride
        anchor_ious = anchor_intersections(self._anchors, gt_boxes_resize)  # b*5*N
        _, anchor_inds = torch.max(anchor_ious, 1)  # b*N

        ## 6. set the corresponding targets for each anchor that is responsible for a ground truth box
        for b in range(batch_size):
            boxes_num = int(num_boxes[b].item())
            for i in range(boxes_num):
                cell_ind = cell_inds[b, i]
                if cell_ind >= hw or cell_ind < 0:
                    print('cell {} over hw {}'.format(cell_ind.item(), hw))
                    continue
                a = anchor_inds[b, i]
                _iou_mask[b, cell_ind, a, :] = self.args['train_args']['object_scale'] * (1 - iou_pred[b, cell_ind, a, :])
                _ious[b, cell_ind, a, :] = ious_reshaped[b, cell_ind, a, i]

                _box_mask[b, cell_ind, a, :] = self.args['train_args']['coord_scale']
                target_boxes[b, i, 2:4] /= self._anchors[a]
                _boxes[b, cell_ind, a, :] = target_boxes[b, i, :]

                _class_mask[b, cell_ind, a, :] = self.args['train_args']['class_scale']
                _classes[b, cell_ind, a, int(gt_boxes[b, i, 4].item())] = 1

        return _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
