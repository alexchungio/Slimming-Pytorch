#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/5 下午3:36
# @ Software   : PyCharm
#-------------------------------------------------------
from __future__ import print_function

import os

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from tools import get_model, save_checkpoint
from configs.cfgs import args
from data.dataset import get_dataset


args.cuda = True if torch.cuda.is_available() else False

if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)


def update_bn(model):
    """
    additional sub-gradient descent on the sparsity-induced penalty term
    :param model:
    :return:
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.add_(args.scale * torch.sign(m.weight.data))  # L1 regularization


def train(model, epoch, data_loader, optimizer, criterion):
    """

    :param model:
    :param epoch:
    :param data_loader:
    :param optimizer:
    :return:
    """
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        optimizer.zero_grad()
        loss = criterion(output, target)
        # pred = output.data.max(1, keepdim=True)[1]
        loss.backward()

        # sparse regularization
        if args.sr:
            update_bn(model)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader), loss.item()))


def test(model, epoch, data_loader):
    """

    :param model:
    :param epoch:
    :param data_loader:
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(data_loader.dataset)
        print('\nTest Epoch : {}[Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)]\n'.format(
             epoch, test_loss, correct, len(data_loader.dataset),
             100. * correct / len(data_loader.dataset)))
        return correct / float(len(data_loader.dataset))

def main():

    # load dataset
    train_dataset, test_dataset = get_dataset()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=False, batch_size=args.test_batch_size)

    # get model
    if args.refine:
        checkpoint = torch.load(args.refine)
        model = get_model(args, cfg=checkpoint['cfg'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = get_model(args)

    if args.cuda:
        model.cuda()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    # lr_scheduler
    milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)

    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(args.resume, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    best_acc = 0.
    for epoch in range(args.start_epoch, args.epochs):

        train(model, epoch, train_loader, optimizer, criterion)
        acc = test(model, epoch, test_loader)
        lr_scheduler.step()
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint(
            state={'epoch': epoch + 1,
                   'arch': args.arch,
                   'depth': args.depth,
                   'state_dict': model.state_dict(),
                   'best_acc': best_acc,
                   'optimizer': optimizer.state_dict(),},
            is_best=is_best,
            filepath=args.checkpoint)

    print("Best accuracy: " + str(best_acc))


if __name__ == "__main__":
    main()