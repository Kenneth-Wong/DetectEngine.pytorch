import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
from PIL import Image

from feeder.utils.minibatch import get_minibatch
from net.utils.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_epoch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, -1).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover_flag = True
            self.leftover = torch.arange(self.num_per_epoch * batch_size, train_size).long()

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_epoch).view(-1, 1) * self.batch_size
        self.rand_num = (rand_num.expand(-1, self.batch_size) + self.range).view(-1).contiguous()
        if self.leftover_flag:
            self.rand_num = torch.cat((self.rand_num, self.leftover), 0)
        return iter(self.rand_num)

    def __len__(self):
        return self.num_data


class RoiFeeder(data.Dataset):
    def __init__(self, roidb, ratio_list, ratio_index, num_classes, training=True, normalize=None, need_bg=True,
                 **args):
        self._roidb = roidb
        self._num_classes = num_classes
        self.ratio_list = ratio_list
        # self.ratio_index is the actual sequnece for sampling, the ratio is sorted in an ascending order.
        self.ratio_index = ratio_index
        self.data_size = len(self.ratio_list)
        self.training = training
        self.normalize = normalize
        self.need_bg = need_bg
        self.args = args

        self.max_num_box = self.args['train_feeder_args']['max_num_gt_boxes']
        self.batch_size = self.args['train_args']['ims_per_batch']

        self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
        num_batch = int(np.ceil(len(ratio_index) / self.batch_size))
        for i in range(num_batch):
            left_idx = i * self.batch_size
            right_idx = min((i + 1) * self.batch_size - 1, self.data_size - 1)

            if ratio_list[right_idx] < 1:
                target_ratio = ratio_list[left_idx]
            elif ratio_list[left_idx] > 1:
                target_ratio = ratio_list[right_idx]
            else:
                target_ratio = 1
            self.ratio_list_batch[left_idx:(right_idx + 1)] = target_ratio

    def __getitem__(self, index):
        if self.training:
            ratio_index = int(self.ratio_index[index])
        else:
            ratio_index = index

        minibatch_db = [self._roidb[ratio_index]]
        blobs = get_minibatch(minibatch_db, self._num_classes, **self.args)
        im_data = blobs['data']
        if not self.need_bg:  ## yolo need rgb image
            im_data = im_data[:, :, :, ::-1]
        data = torch.from_numpy(im_data.copy())  # 1 * h * w * 3
        im_info = torch.from_numpy(blobs['im_info'])  # 1 * 3
        data_height, data_width = data.size(1), data.size(2)

        if self.training:
            np.random.shuffle(blobs['gt_boxes'])
            gt_boxes = blobs['gt_boxes']
            if not self.need_bg:
                gt_boxes[:, -1] -= 1
            gt_boxes = torch.from_numpy(gt_boxes)  # N * 5

            ratio = self.ratio_list_batch[index]

            if self._roidb[ratio_index]['need_crop']:
                if ratio < 1:
                    # in this group, the max ratio is less than 1, and this image has need_crop flag(ratio<0.5),
                    # it means that w << h, crop h
                    min_y = int(torch.min(gt_boxes[:, 1]))
                    max_y = int(torch.max(gt_boxes[:, 3]))
                    box_region = max_y - min_y + 1

                    trim_size = int(np.floor(data_width / ratio))
                    if trim_size > data_height:
                        trim_size = data_height

                    if min_y == 0:
                        y_s = 0
                    else:
                        if box_region - trim_size < 0:
                            y_s_min = max(max_y - trim_size, 0)
                            y_s_max = min(min_y, data_height - trim_size)
                            if y_s_min == y_s_max:
                                y_s = y_s_min
                            else:
                                y_s = np.random.choice(range(y_s_min, y_s_max))
                        else:
                            y_s_add = int((box_region - trim_size) / 2)
                            if y_s_add == 0:
                                y_s = min_y
                            else:
                                y_s = np.random.choice(range(min_y, min_y + y_s_add))
                    data = data[:, y_s:(y_s + trim_size), :, :]
                    gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                    gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)
                    gt_boxes[:, 1].clamp_(0, trim_size - 1)
                    gt_boxes[:, 3].clamp_(0, trim_size - 1)
                else:
                    min_x = int(torch.min(gt_boxes[:, 0]))
                    max_x = int(torch.max(gt_boxes[:, 2]))
                    box_region = max_x - min_x + 1

                    trim_size = int(np.floor(data_height * ratio))
                    if trim_size > data_width:
                        trim_size = data_width

                    if min_x == 0:
                        x_s = 0
                    else:
                        if box_region - trim_size < 0:
                            x_s_min = max(max_x - trim_size, 0)
                            x_s_max = min(min_x, data_width - trim_size)
                            if x_s_min == x_s_max:
                                x_s = x_s_min
                            else:
                                x_s = np.random.choice(range(x_s_min, x_s_max))
                        else:
                            x_s_add = int((box_region - trim_size) / 2)
                            if x_s_add == 0:
                                x_s = min_x
                            else:
                                x_s = np.random.choice(range(min_x, min_x + x_s_add))
                    data = data[:, :, x_s:(x_s + trim_size), :]
                    gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                    gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                    gt_boxes[:, 0].clamp_(0, trim_size - 1)
                    gt_boxes[:, 2].clamp_(0, trim_size - 1)

            if ratio < 1:
                # width < height, it means widths are the same, e.g. 600
                padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), data_width, 3).zero_()
                padding_data[:data_height, :, :] = data[0]
                im_info[0, 0] = padding_data.size(0)
            elif ratio > 1:
                # width > height, it means heights are the same, e.g. 600
                padding_data = torch.FloatTensor(data_height, int(np.ceil(data_height * ratio)), 3).zero_()
                padding_data[:, :data_width, :] = data[0]
                im_info[0, 1] = padding_data.size(1)
            else:
                # it means that height and width is similar, one of them is 600
                trim_size = min(data_height, data_width)
                padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
                padding_data = data[0][:trim_size, :trim_size, :]
                gt_boxes[:, :4].clamp(0, trim_size - 1)
                im_info[0, 0] = trim_size
                im_info[0, 1] = trim_size

            not_keep = (gt_boxes[:, 0] == gt_boxes[:, 2]) | (gt_boxes[:, 1] == gt_boxes[:, 3])
            keep = torch.nonzero(not_keep == 0).view(-1)

            gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
            if keep.numel() != 0:
                gt_boxes = gt_boxes[keep]
                num_boxes = min(gt_boxes.size(0), self.max_num_box)
                gt_boxes_padding[:num_boxes, :] = gt_boxes[:num_boxes]
            else:
                num_boxes = 0

            padding_data = padding_data.permute(2, 0, 1).contiguous()
            im_info = im_info.view(3)
            return padding_data, im_info, gt_boxes_padding, num_boxes
        else:
            data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
            im_info = im_info.view(3)
            gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
            num_boxes = 0
            return data, im_info, gt_boxes, num_boxes

    def __len__(self):
        return len(self._roidb)
