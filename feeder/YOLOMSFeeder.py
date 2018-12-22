import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
from PIL import Image

from feeder.utils.minibatch import get_size_minibatch
from net.utils.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time


class sampler(Sampler):
    def __init__(self, train_size, batch_size, group_size):
        self.num_data = train_size
        self.num_batch = int(train_size / batch_size)  # drop the last one
        self.batch_size = batch_size
        self.group_size = group_size  ## num of batch each group contains
        self.num_group_sample = self.group_size * self.batch_size
        self.range = torch.arange(0, self.num_group_sample).view(1, -1).long()
        self.num_group = int(self.num_batch / self.group_size)
        self.leftover_flag = False
        if self.num_batch % self.group_size:
            self.leftover_flag = True
            self.leftover = torch.arange(self.num_group * self.num_group_sample,
                                         self.num_batch * self.batch_size).long()

    def __iter__(self):
        rand_num = torch.randperm(self.num_group).view(-1, 1) * self.num_group_sample
        self.rand_num = (rand_num.expand(-1, self.num_group_sample) + self.range).view(-1).contiguous()
        ## shuffle inside each group again
        for i in range(self.num_group):
            self.rand_num[(i * self.num_group_sample):((i + 1) * self.num_group_sample)] = torch.randperm(
                self.num_group_sample) + i * self.num_group_sample
        if self.leftover_flag:
            self.rand_num = torch.cat((self.rand_num, self.leftover), 0)
        return iter(self.rand_num)

    def __len__(self):
        return self.num_batch * self.batch_size


class YOLOMSFeeder(data.Dataset):
    def __init__(self, roidb, num_classes, training=True, normalize=None, need_bg=False, **args):
        self._roidb = roidb
        self._num_classes = num_classes
        self.data_size = len(roidb)
        self.training = training
        self.normalize = normalize
        self.need_bg = need_bg
        self.args = args

        self.max_num_box = self.args['train_feeder_args']['max_num_gt_boxes']
        self.batch_size = self.args['train_args']['ims_per_batch']
        self.group_size = self.args['train_args']['group_size']  ## 10

        self.num_batch = int(np.ceil(self.data_size / self.batch_size))

    def reset_size_batch(self, size_index=None):
        if self.training:
            print("Random the size index again...")
            self.size_batch_list = torch.Tensor(self.data_size).zero_().long()
            num_var_size = int(np.ceil(self.num_batch / self.group_size))
            size = np.random.randint(0, len(self.args['mix_args']['input_size']), num_var_size)
            for i, s in enumerate(size):
                left_idx = i * self.batch_size * self.group_size
                right_idx = min((i + 1) * self.batch_size * self.group_size, self.data_size - 1)
                self.size_batch_list[left_idx:(right_idx + 1)] = int(s)
            print('Done.')
            print('len: {}, size: {}'.format(num_var_size, size))
        else:
            assert size_index is not None, "Size index must be provided during testing."
            self.size_batch_list = torch.Tensor(self.data_size).fill_(size_index).long()

    def __getitem__(self, index):
        minibatch_db = [self._roidb[index]]
        size_index = self.size_batch_list[index]
        blobs = get_size_minibatch(minibatch_db, self.args['mix_args']['input_size'], size_index, max_num_box=self.max_num_box,
                                   is_training=self.training)
        im_data, im_info, gt_boxes, num_boxes = blobs

        data = torch.from_numpy(im_data.copy()).permute(2, 0, 1).contiguous()  # 3*h*w
        im_info = torch.from_numpy(im_info).float().view(3)  #  3
        gt_boxes = torch.from_numpy(gt_boxes).float()
        return data, im_info, gt_boxes, num_boxes

    def __len__(self):
        return len(self._roidb)
