#!/usr/bin/env python
import argparse
import sys
# torchlight
import torchlight
from torchlight import import_class
import os
import torch

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')
    # region register processor yapf: disable
    processors = dict()
    processors['det_frcnn'] = import_class('processor.detection.frcnn.frcnn_Processor')
    processors['det_yolov2'] = import_class('processor.detection.yolov2.yolov2_Processor')
    # processors['sg_imp'] = import_class('processor.sg.imp.imp_Porcessor')
    # processors['demo'] = import_class('processor.demo.Demo')
    # endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    p.start()
