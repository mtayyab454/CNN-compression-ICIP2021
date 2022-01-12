from __future__ import print_function

import sys
sys.path.insert(1, '../')
from models.models import get_cifar_models

import os
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
from base_code.basisModel import basisModel

from utils import Logger, get_time_str, create_dir, backup_code
from trainer import testing_loop, training_loop

from base_code.basisModel import display_stats as base_display_stats

def display_stats(basis_model, model, exp_name, input_size=[32, 32], count_relu=False):
    return base_display_stats(basis_model, model, exp_name, input_size, count_relu)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

model_names = ['resnet56']

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets

parser.add_argument('--jobid', type=str, default='test')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet56',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet56)')

parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Compression options
parser.add_argument('--l1_weight', default=0, type=float)
parser.add_argument('--l2_weight', default=0.1, type=float)
parser.add_argument('--t', default=0.63, type=float, help='value of compression ratio (between 0 and 1)')
parser.add_argument("--add_bn", type=str2bool, nargs='?', const=True, default=True, help="Use batchnorm between basis filters and 1by1 convolutions.")
# Optimization options
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--schedule', type=int, nargs='+', default=[20, 40],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--logs', default='logs', type=str, metavar='PATH',
                    help='path to save the training logs (default: logs)')
# Architecture
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

def main():
    print(args)
    exp_name = args.jobid + '_' + args.arch
    checkpoint_dir = os.path.join(args.checkpoint, exp_name)
    create_dir([checkpoint_dir, args.logs])

    model = get_cifar_models(args.arch, args.dataset, pretrained=True)
    basis_model = basisModel(model, use_weights=True, add_bn=args.add_bn, trainable_basis=True, replace_fc=False, sparse_filters=False)

    basis_model.cuda()
    model.cuda()

    # basis_model, model = get_models(model_name, dataset_name, sparse_filters=False, pretrained=True, add_bn=add_bn)
    basis_model.update_channels(args.t)
    stats = display_stats(basis_model, model, args.arch + '-' + args.dataset)

    logger = Logger(dir_path=args.logs, fname=exp_name,
                    keys=['time', 'acc1', 'acc5', 'loss', 'ce_loss', 'l1_loss', 'l2_loss'])
    logger.one_time({'stats': stats, 'comments': args.arch + '-' + args.dataset})
    logger.set_names(['lr', 'train_stats', 'test_stats'])

    testing_loop(basis_model, args.dataset)

    training_loop(model=basis_model, logger=logger, schedule=args.schedule, training_opts=[args.lr, args.momentum, args.weight_decay, args.gamma],
                  loss_weights=[args.l1_weight, args.l2_weight], dataset_name=args.dataset, args=args, save_best=True)

if __name__ == '__main__':
    main()