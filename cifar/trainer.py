import os
import torch
import time
import torch.nn as nn
import torch.optim as optim

from cifar.utils import AverageAccumulator, VectorAccumulator, accuracy, Progressbar, adjust_learning_rate, get_num_parameters
from cifar.datasets import get_cifar_data
from base_code.basis_loss import basisCombinationLoss

def train(trainloader, model, optimizer, criterion, keys):
    print('Training...')
    model.train()

    accumulator = VectorAccumulator(keys)
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(Progressbar(trainloader)):
        # measure data loading time
        # print(batch_idx)
        inputs = inputs.cuda()
        targets = targets.cuda()

        # compute output
        outputs = model(inputs)
        losses = criterion(model, outputs, targets)
        # losses.update(loss.item())

        # prec1 = sum(model_pred.squeeze(1) == targets)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        # gt_acc.update(prec1.item())

        optimizer.zero_grad()
        losses[0].backward()
        optimizer.step()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        accumulator.update( [(time.time() - end), prec1.item(), prec5.item()] + [l.item() for l in losses] )
        end = time.time()

    return accumulator.avg

def test(testloader, model, criterion, keys):
    print('Testing...')
    # switch to evaluate mode
    model.eval()

    accumulator = VectorAccumulator(keys)
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(Progressbar(testloader)):

        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
        # loss = criterion(outputs, targets)
        losses = criterion(model, outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        accumulator.update( [(time.time() - end), prec1.item(), prec5.item()] + [l.item() for l in losses] )

        end = time.time()

    return accumulator.avg

def testing_loop(model, dataset_name):
    criterion = basisCombinationLoss(0, 0, False)
    _, testloader, num_classes = get_cifar_data(dataset_name, split='test', batch_size=100, num_workers=0)
    test_stats = test(testloader, model, criterion, ['time', 'acc1', 'acc5', 'loss', 'ce_loss', 'l1_loss', 'l2_loss'])
    print('\nTest loss: %.4f \nVal accuracy: %.2f%%' % (test_stats[3], test_stats[1]))

def training_loop(model, logger, schedule, training_opts, loss_weights, dataset_name, args, save_best=False):
    criterion = basisCombinationLoss(loss_weights[0], loss_weights[0], False)

    criterion.cuda()

    ###################### Initialization ###################
    # Unpack inputs
    num_epochs, schedule = schedule[-1], schedule[:-1]
    lr, momentum, weight_decay, gamma = training_opts

    # Load data
    _, trainloader, num_classes = get_cifar_data(dataset_name, split='train', batch_size=args.train_batch, num_workers=args.workers)
    _, testloader, num_classes = get_cifar_data(dataset_name, split='test', batch_size=args.test_batch, num_workers=args.workers)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    num_param = get_num_parameters(model)

    print('    Total params: %.2fM' % (num_param / 1000000.0))
    logger.one_time({'num_param': num_param})

    ###################### Main Loop ########################
    best_acc = 0
    for epoch in range(num_epochs):
        lr = adjust_learning_rate(optimizer, lr, epoch, schedule, gamma)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, num_epochs, lr))

        train_stats = train(trainloader, model, optimizer, criterion, logger.keys)
        test_stats = test(testloader, model, criterion, logger.keys)

        torch.save(model.state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '.pth'))

        if best_acc < test_stats[1]:
            best_acc = test_stats[1]
            if save_best:
                torch.save(model.state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '_best.pth'))

        print('\nKeys: ', logger.keys)
        print('Training: ', train_stats)
        print('Testing: ', test_stats)
        print('Best Acc: ', best_acc)

        logger.append([lr, train_stats, test_stats])

    return model