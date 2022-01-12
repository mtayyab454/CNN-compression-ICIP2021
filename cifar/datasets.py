import os
import torch
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np
from PIL import Image

class CIFAR_Sub_Class(data.Dataset):
    def __init__(self, CIFAR_obj, classes):

        self.transform = CIFAR_obj.transform
        self.target_transform = CIFAR_obj.target_transform
        self.train = CIFAR_obj.train
        self.classes = classes

        # Create sub set of data from classes

        labels = CIFAR_obj.train_labels if self.train else CIFAR_obj.test_labels
        data = CIFAR_obj.train_data if self.train else CIFAR_obj.test_data

        idx, _ = self.get_match_index(labels, classes)

        data = data[idx, :, :, :]
        temp = np.array([labels])
        labels = temp[0][idx].tolist()

        # now load the picked numpy arrays
        if self.train:
            self.train_labels = labels
            self.train_data = data
        else:
            self.test_labels = labels
            self.test_data = data

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_match_index(self, target, labels):
        label_indices = []
        new_targets = []

        for i in range(len(target)):
            if target[i] in labels:
                label_indices.append(i)
                new_targets.append(np.where(np.array(labels) == target[i])[0].item())

        return label_indices, new_targets

def get_cifar_data(data_set, split, batch_size=100, num_workers=4):
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
        trainset = dataloader(root=os.path.join('.','../../data/CIFAR'), train=True, download=True, transform=transform_train)
        testset = dataloader(root=os.path.join('.','../../data/CIFAR'), train=False, download=False, transform=transform_test)

    elif data_set in ['CIFAR100', 'cifar100']:
        dataloader = datasets.CIFAR100
        num_classes = 100
        trainset = dataloader(root=os.path.join('.','../../data/CIFAR'), train=True, download=True, transform=transform_train)
        testset = dataloader(root=os.path.join('.','../../data/CIFAR'), train=False, download=False, transform=transform_test)

    if split == 'train':
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return trainset, trainloader, num_classes
    elif split == 'test':
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return testset, testloader, num_classes

def get_cifar_sub_class(dataset, classes, split, batch_size=100, num_workers=4):
    temp_set, _, _ = get_cifar_data(dataset, split, batch_size, num_workers)
    dset = CIFAR_Sub_Class(temp_set, classes)

    if split == 'train':
        dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    elif split == 'test':
        dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dset, dloader, len(classes)