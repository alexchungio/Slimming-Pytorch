#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : dataset.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/5 下午3:39
# @ Software   : PyCharm
#-------------------------------------------------------

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from configs.cfgs import args




train_transforms = transforms.Compose([transforms.Pad(4),
                                      transforms.RandomCrop(32),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

test_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


def get_dataset():
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10('../cifar10', train=True, download=True,
                                        transform=train_transforms)
        test_dataset =  datasets.CIFAR10('../cifar10', train=False, transform=test_transforms)


    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100('./cifar100', train=True, download=True,
                                         transform=train_transforms)
        test_dataset = datasets.CIFAR100('./cifar100', train=False, transform=test_transforms)
    else:
        raise AttributeError

    return train_dataset, test_dataset



def main():
    train_dataset, test_dataset = get_dataset()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=False, batch_size=args.test_batch_size)

    print(len(train_loader))
    print(len(test_loader))


if __name__ == "__main__":
    main()