import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
from datetime import datetime

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class SubClassAccuracy(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, num_classes):
        # num_classes is total number of classes in the data-set, eg for CIFAR100 it should be 100
        self.labels = []
        self.pred = []
        self.num_classes = num_classes

    def subclass_accuracy(self, class_idx):
        # class_idx is a list of classes for which we need to claculate accuracy separately, eg [10, 15, 31, 38, 83]
        _, class_acc_vec = self.get_accuracy()
        subclass_acc_vec = []
        for c in class_idx:
            subclass_acc_vec.append(class_acc_vec[c])

        subclass_acc = sum(subclass_acc_vec)/len(class_idx)

        return subclass_acc, subclass_acc_vec

    def update(self, labels, output):
        self.labels.extend(labels)
        self.pred.extend(output)

    def get_accuracy(self):
        class_correct_vec = list(0. for i in range(self.num_classes))
        class_total_vec = list(0. for i in range(self.num_classes))
        class_acc_vec = list(0. for i in range(self.num_classes))

        c = np.array(self.labels) == np.array(self.pred)
        for i in range(len(self.labels)):
            label = self.labels[i]
            class_correct_vec[label] += c[i]
            class_total_vec[label] += 1

        for i in range(self.num_classes):
            if class_total_vec[i] != 0:
                class_acc_vec[i] = round(100 * class_correct_vec[i] / class_total_vec[i], 2)
            # print('Accuracy of %2d : %2d %%' % (i, class_acc[i]))

        acc = round(sum(class_correct_vec)/sum(class_total_vec), 2)

        return acc, class_acc_vec

class AverageAccumulator(object):
    """Computes and stores the averages and current values for a set
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = []
        self.avg = 0

    def update(self, val, n=1):
        temp_val = [val for i in range(n)]
        self.val.extend(temp_val)
        self.avg = sum(self.val) / len(self.val)

class VectorAccumulator(object):
    """Stores every value for a set of elements
    """
    def __init__(self, keys):

        self.n = len(keys)
        self.keys = keys
        self.reset()

    def reset(self):
        self.val = [0.0 for i in range(self.n)]
        self.count = 0
        self.avg = [0.0 for i in range(self.n)]

    def update(self, val):

        assert(len(val) == self.n)

        self.count += 1
        for i, v in enumerate(val):
            self.val[i] += v

        for i, v in enumerate(val):
            self.avg[i] = self.val[i] / self.count

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Progressbar:
    def __init__(self, iterr, step=20, prefix=' -> '):
        self.iter = iter(iterr)
        self.max = len(iterr)
        self.counter = 0
        self.step = step
        self.milestone = step
        self.prefix = prefix

    def __iter__(self):
        return self

    def __next__(self):
        self.update()
        return self.iter.__next__()

    def update(self):
        progress = self.counter/self.max*100.0
        if progress >= self.milestone:
            print(self.prefix + 'Progress: %.f done!' %(progress) )
            self.milestone = self.milestone + self.step

        self.counter = self.counter+1

class Logger(object):
    '''Save training process to log file with simple plot function.'''

    def __init__(self, dir_path, fname, keys):
        dir_path = os.path.join(dir_path, fname)
        if os.path.exists(dir_path) == False:
            os.makedirs(dir_path)
        fpath = os.path.join(dir_path, fname + '.mat')
        self.dir_path = dir_path
        self.fname = fname
        self.fpath = fpath
        self.numbers = {}
        self.keys = keys

        self.one_time({'keys':keys})

    def one_time(self, new_dict):
        self.numbers.update(new_dict)

    def set_names(self, names):
        self.names = names
        for _, name in enumerate(self.names):
            self.numbers[name] = []

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.numbers[self.names[index]].append(num)

        sio.savemat(self.fpath, self.numbers)

def adjust_learning_rate(optimizer, lr, epoch, schedule, gamma):
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr

def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_num_cofficients(model):
    return sum(p.numel() for p in model.basis_parameters())

def create_dir(dir_paths): # dir_paths: relative paths of the directories to create.

    if isinstance(dir_paths, list):
        for dp in dir_paths:
            if os.path.exists(dp) == False:
                os.makedirs(dp)
    else:
        if os.path.exists(dir_paths) == False:
            os.makedirs(dir_paths)

def backup_code(backup_folder, code_files): # code_files: relative paths of the code file to backup in a list
    if os.path.exists(backup_folder) == False:
        os.makedirs(backup_folder)

    if isinstance(code_files, list):
        for cf in code_files:
            command = 'cp ' + cf + ' ' + backup_folder
            os.system(command)
    else:
        command = 'cp ' + code_files + ' ' + backup_folder
        os.system(command)

def get_time_str():
    now = datetime.now()

    current_time = now.strftime("%Y.%m.%d-%H.%M.%S")
    return current_time

def replace_none(args_dict):

    for k in args_dict.keys():
        if args_dict[k] is None:
            args_dict[k] = []

    return args_dict

# pbar = ProgressBar()
# for i in progressbar(range(155)):
#     # print(i)
#     pass

# create_dir('test')
# backup_code('test', ['utils.py', 'trainval.py'])
# v = VectorAccumulator(3, ['asd'])
# v.update([1, 2, 3])
# v.update([3, 2, 1])
# print(v.avg)