import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
def get_cifar_data(data_set, data_path, split, batch_size=100, num_workers=4):
    print('==> Preparing dataset %s' % data_set)
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    # ])
    #
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    # ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if data_set in ['CIFAR10', 'cifar10']:
        dataloader = datasets.CIFAR10
        num_classes = 10
        trainset = dataloader(root=os.path.join('.',data_path), train=True, download=True, transform=transform_train)
        testset = dataloader(root=os.path.join('.',data_path), train=False, download=False, transform=transform_test)

    elif data_set in ['CIFAR100', 'cifar100']:
        dataloader = datasets.CIFAR100
        num_classes = 100
        trainset = dataloader(root=os.path.join('.',data_path), train=True, download=True, transform=transform_train)
        testset = dataloader(root=os.path.join('.',data_path), train=False, download=False, transform=transform_test)

    if split == 'train':
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return trainset, trainloader, num_classes
    elif split == 'test':
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return testset, testloader, num_classes