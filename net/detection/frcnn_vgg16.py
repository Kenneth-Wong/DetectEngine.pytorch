import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from ..frcnn import fasterRCNN
import pdb


class vgg16(fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False, **args):
        self.args = args
        self.model_path = args['pretrained_weight']
        if pretrained:
            assert self.model_path is not None, 'Pretrained model is not provided.'
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        fasterRCNN.__init__(self, classes, class_agnostic, **args)

    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})
            print('done')

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])  # remove fc8, 1000 classes

        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])  # remove maxpool

        # Fix layer conv1, conv2:
        for layer in range(10):
            for p in self.RCNN_base[layer].parameters(): p.require_grad = False

        self.RCNN_top = vgg.classifier

        self.RCNN_cls_score = nn.Linear(4096, self.num_classes)

        nc_box = 4 if self.class_agnostic else 4 * self.num_classes
        self.RCNN_bbox_pred = nn.Linear(4096, nc_box)

    def _head_to_tail(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)
        return fc7
