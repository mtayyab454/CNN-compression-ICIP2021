from __future__ import print_function

import sys
sys.path.insert(1, '../')

import os
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import cifar.models as models
from cifar.utils import Logger, create_dir, backup_code
from cifar.trainer import testing_loop, training_loop

from base_code.basis_model import replace_conv2d_with_basisconv2d, trace_model, get_basis_channels_from_t, display_stats

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets

parser.add_argument('--jobid', type=str, default='test')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet56',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet56)')

parser.add_argument("--baseline", type=str2bool, nargs='?', const=True, default=False, help="Set true to train the baseline model, without any compression")
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('--data-path', default='../../data/CIFAR', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Compression options
parser.add_argument('--l1-weight', default=0, type=float)
parser.add_argument('--l2-weight', default=0.001, type=float)
# Add description of compress_rate
parser.add_argument('--compress-rate', type=str, default='0.50', help='compress rate of each conv')
parser.add_argument("--add-bn", type=str2bool, nargs='?', const=True, default=True, help="Use batchnorm between basis filters and 1by1 convolutions.")
# Optimization options
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 90],
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
parser.add_argument('-c', '--checkpoint', default='cifar/checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--logs', default='cifar/logs', type=str, metavar='PATH',
                    help='path to save the training logs (default: logs)')
# Architecture
# Miscs
parser.add_argument('--manual-seed', type=int, default=None, help='manual seed')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset and set num_classes
if args.dataset == 'cifar10':
    args.num_classes = 10
elif args.dataset == 'cifar100':
    args.num_classes = 100
else:
    raise ValueError(f"Invalid dataset specified: {args.dataset}. Supported values are 'cifar10' and 'cifar100'.")

# Validate compression parameters
if args.compress_rate[0] == ':':
    args.compress_rate = [float(x) for x in args.compress_rate[2:-1].split(',')]
else:
    args.compress_rate = float(args.compress_rate)

print('compress_rate:', args.compress_rate)


# Random seed
if args.manual_seed is None:
    args.manual_seed = random.randint(1, 10000)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

def main():
    print(args)
    exp_name = args.jobid + '_' + args.arch
    checkpoint_dir = os.path.join(args.checkpoint, exp_name)
    create_dir([checkpoint_dir, args.logs])

    model = models.__dict__[args.arch](num_classes=args.num_classes)

    if args.baseline is True:
        basis_model = model
        stats = 'Baseline training'
    else:
        num_conv, num_linear, in_channels, out_channels, basis_channels, layer_type = trace_model(model)
        if isinstance(args.compress_rate, list):
            assert len(args.compress_rate) == num_conv, "Number of values of t must equal to number of convolution layers"
        else:
            args.compress_rate = [args.compress_rate] * num_conv
        _, _, basis_channels = get_basis_channels_from_t(model, args.compress_rate)

        basis_model = replace_conv2d_with_basisconv2d(model, basis_channels, [args.add_bn] * num_conv)

        stats = display_stats(basis_model, model, args.arch + '-' + args.dataset, [3, 32, 32])

    basis_model.cuda()

    logger = Logger(dir_path=args.logs, fname=exp_name,
                    keys=['time', 'acc1', 'acc5', 'loss', 'ce_loss', 'l1_loss', 'l2_loss'])
    logger.one_time({'stats': stats, 'seed':args.manual_seed, 'comments': args.arch + '-' + args.dataset})
    logger.set_names(['lr', 'train_stats', 'test_stats'])

    testing_loop(basis_model, args)

    training_loop(model=basis_model, logger=logger, args=args, save_best=True)

if __name__ == '__main__':
    main()