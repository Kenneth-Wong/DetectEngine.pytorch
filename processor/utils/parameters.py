import argparse
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class


def get_basic_parser(add_help=False):
    # region arguments yapf: disable
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(add_help=add_help, description='IO Processor')

    parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
    parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

    # processor
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
    parser.add_argument('--device', type=int, default=0, nargs='+',
                        help='the indexes of GPUs for training or testing')

    # visulize and debug
    parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
                        help='the name of weights which will be ignored in the initialization')
    # endregion yapf: enable

    return parser


def get_detbase_parser(add_help=False):
    # region arguments yapf: disable
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(add_help=add_help, description='Base Processor')

    parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
    parser.add_argument('-r', '--data_root', default='./data', help='the work folder for storing results')
    parser.add_argument('-d', '--data_dir', default='./data/VOCdevkit2007', help='the data folder for storing data')
    parser.add_argument('-c', '--config', default=None, help='path to the configuration file')
    parser.add_argument('-s', '--session', default=1, type=int, help='the session of this experiment')
    parser.add_argument('--fix_random', default=True, type=str2bool, help='whether to fix the random seed')
    parser.add_argument('--snapshot_prefix', default='frcnn_vgg16_voc', help='snapshot prefix')
    parser.add_argument('--resume', type=str2bool, default=False, help='resume from checkpoint')
    parser.add_argument('--use_tfboard', type=str2bool, default=False, help='whether to use tensorboardX')

    # opt
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--save_result', type=str2bool, default=False,
                        help='if true, the output of the model will be stored')
    parser.add_argument('--opt_args', type=DictAction, default=dict(),
                        help='arguments of optimizer for training')

    # device
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to use gpu')
    parser.add_argument('--device', type=int, default=0, nargs='+',
                        help='the indexes of GPUs for training or testing')

    # visulize and debug
    parser.add_argument('--log_interval', type=int, default=10,
                        help='the interval for printing messages (#iteration)')
    parser.add_argument('--save_interval', type=int, default=4,
                        help='the interval for storing models (#iteration)')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='the interval for evaluating models (#iteration)')
    parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
    parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--pavi_log', type=str2bool, default=False, help='logging on pavi or not')

    # feeder & dataset
    parser.add_argument('--train_dataset', default='voc_2007_trainval', help='training dataset')
    parser.add_argument('--val_dataset', default='voc_2007_test', help='validating dataset')
    parser.add_argument('--test_dataset', default='voc_2007_test', help='testing dataset')
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--sampler', default=None, help='data sampler will be used')
    parser.add_argument('--num_worker', type=int, default=8, help='the number of worker per gpu for data loader')
    parser.add_argument('--debug', action="store_true", help='less data, faster loading')
    parser.add_argument('--train_feeder_args', action=DictAction, default=dict(),
                        help='the arguments of data loader for training')
    parser.add_argument('--test_feeder_args', action=DictAction, default=dict(),
                        help='the arguments of data loader for test')

    # model about train and test
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--train_args', action=DictAction, default=dict(), help='the arguments of training')
    parser.add_argument('--test_args', action=DictAction, default=dict(), help='the arguments of testing')
    parser.add_argument('--mix_args', action=DictAction, default=dict(), help='the arguments for both')
    parser.add_argument('--resnet_args', action=DictAction, default=dict(), help='the arguments of resnet')
    parser.add_argument('--pretrained_weight', default=None, help='the weights for network initialization')
    parser.add_argument('--checkpoint', default=None, help='the weights for finetuning or test')
    parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
                        help='the name of weights which will be ignored in the initialization')
    # endregion yapf: enable

    return parser
