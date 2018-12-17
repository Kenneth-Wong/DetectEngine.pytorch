import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import time
from ..yolov2 import yolov2
from ..yolov2 import _make_layers
from net.utils.net_utils import Conv2d_BatchNorm


def darknet19():
    net_cfgs = [
        # conv1s
        [(32, 3)],
        ['M', (64, 3)],
        ['M', (128, 3), (64, 1), (128, 3)],
        ['M', (256, 3), (128, 1), (256, 3)],
        ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)],
        # conv2
        ['M', (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)],
    ]
    conv1s, c1 = _make_layers(3, net_cfgs[0:5])
    conv2, c2 = _make_layers(c1, net_cfgs[5])
    return conv1s, c1, conv2, c2


class yolov2_darknet19(yolov2):
    """
    darknet19-basenet, dropping the last conv layer of darknet19 trained on imagenet,
    which contains conv 1s(feat stride 16) + conv2(feat stride 32, totally)
    """

    def __init__(self, classes, pretrained=False, fix_bn=False, **args):
        self.args = args
        self.model_path = args['pretrained_weight']
        if pretrained:
            assert self.model_path is not None, 'Pretrained model is not provided.'
        self.pretrained = pretrained
        self.dout_base_model = 1024  # c2
        self.dout_middle_model = 512  # c1
        self.fix_bn = fix_bn

        yolov2.__init__(self, classes, **args)

    def _init_modules(self):
        conv1s, c1, conv2, c2 = darknet19()
        if self.pretrained:
            print("Loading pretrained weights from %s" % self.model_path)
            conv1s = self._load_from_npz(conv1s, start=0, num_conv=13)
            conv2 = self._load_from_npz(conv2, start=13, num_conv=5)
            print('Done.')

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False
        if self.fix_bn:
            conv1s.apply(set_bn_fix)
            conv2.apply(set_bn_fix)

        # normally, yolov2_base is nn.sequential module,
        # however, darknet19 needs residual connection between conv1s and conv3,
        # thus we return a list
        self.yolov2_base = [conv1s, conv2]
        self.conv1s = conv1s
        self.conv2 = conv2

    def _load_from_npz(self, modules, start, num_conv=None):
        dest_src = {'conv.weight': 'kernel', 'conv.bias': 'biases',
                    'bn.weight': 'gamma', 'bn.bias': 'biases',
                    'bn.running_mean': 'moving_mean',
                    'bn.running_var': 'moving_variance'}
        params = np.load(self.model_path)
        own_dict = modules.state_dict()
        keys = list(own_dict.keys())

        if num_conv is not None:
            num_conv_load = min(int(len(keys)//5), num_conv)  ## Conv2d-BN module num
        else:
            num_conv_load = len(keys)

        for i in range(start, start+num_conv_load):
            for key in keys[((i-start)*5):((i-start)*5+5)]:
                list_key = key.split('.')
                ptype = dest_src['{}.{}'.format(list_key[-2], list_key[-1])]
                src_key = '{}-convolutional/{}:0'.format(i, ptype)
                print((src_key, own_dict[key].size(), params[src_key].shape))
                param = torch.from_numpy(params[src_key])
                if ptype == 'kernel':
                    param = param.permute(3, 2, 0, 1) ## from kh, kw, in_c, out_c to out_c, in_c, kh, kw
                own_dict[key].copy_(param)
        return modules





