import torch
import torch.nn as nn
import numpy as np
import cv2

class _YOLOPreprocessLayer(nn.Module):
    """
    Preprocess the input image: resize the image to the expected input size and
    add color distraction, rescale the color space to [0, 1]
    """

    def __init__(self, **args):
        super(_YOLOPreprocessLayer, self).__init__()
        self._input_size = args['mix_args']['input_size']
        self.args = args

    def forward(self, im_data, gt_boxes, size_index, jitter=0.1):
        phase = self.args['phase']
        batch_size = im_data.size(0)
        h_ori, w_ori = im_data.size(2), im_data.size(3)
        if phase == 'train':
            # jitter the color space
            t = torch.rand(3) * 2 - 1  ## [-1, 1)
            im_data_new = im_data * (1 + t.view(1, 3, 1, 1).cuda() * jitter)
            mx = 255. * (1 + jitter)
            up = torch.rand(1).cuda() * 2 - 1
            im_data_new = torch.pow(im_data_new / mx, 1. + up * 0.5)
        else:
            im_data_new = im_data / 255.

        #  reshape
        W, H = self._input_size[size_index]

        # transform to numpy value
        im_data_np = im_data_new.data.cpu().numpy().transpose(0, 2, 3, 1)  ## b*h_ori*w_ori*3
        im_data_resize_np = np.zeros((batch_size, 3, H, W))

        for i in range(batch_size):
            im_data_resize_np[i] = cv2.resize(im_data_np[i], (W, H)).transpose(2, 0, 1)
            if phase == 'train':
                gt_boxes[i, :, :4][:, 0::2] *= (float(W) / float(w_ori))
                gt_boxes[i, :, :4][:, 1::2] *= (float(H) / float(h_ori))

        # transform to tensor
        im_data_pt = torch.from_numpy(im_data_resize_np).type(torch.FloatTensor).cuda()
        return im_data_pt, gt_boxes







