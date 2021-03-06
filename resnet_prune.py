#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : resnet_prune.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/5 下午8:12
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import argparse
import torch
import torch.nn as nn

from models import channel_selection
from data.dataset import get_dataset
from tools import get_model
from configs.cfgs import args

# Prune settings
args.cuda = torch.cuda.is_available()



def get_bn_info(model):

    # calculate number of bn
    num_bn_channel = 0

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            num_bn_channel += m.weight.data.shape[0]

    bn = torch.zeros(num_bn_channel)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    threshold_index = int(num_bn_channel * args.percent)
    prune_threshold = y[threshold_index]

    return  num_bn_channel, prune_threshold


def pre_prune(model, threshold, total):
    pruned = 0
    cfg = []
    cfg_mask = []

    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(threshold).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = pruned / total

    return model, cfg, cfg_mask, pruned_ratio


def execute_pruned(pre_pruned_model, pruned_model, cfg_mask):
    """

    :param pre_pruned_model:
    :param pruned_model:
    :param cfg_mask:
    :return:
    """
    old_modules = list(pre_pruned_model.modules())
    new_modules = list(pruned_model.modules())

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    first_conv = True

    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            if isinstance(old_modules[layer_id + 1], channel_selection):
                # If the next layer is the channel selection layer, then the current batch normalization layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the mask parameter `indexes` for the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
                continue

        elif isinstance(m0, nn.Conv2d):
            if first_conv:
                # We don't change the first convolution layer.
                m1.weight.data = m0.weight.data.clone()
                first_conv = False
                continue
            if isinstance(old_modules[layer_id - 1], channel_selection):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))

                # If the last layer is channel selection layer, then we don't change the number of output channels of the current
                # convolutional layer.
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                m1.weight.data = w1.clone()
                continue

        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    return pruned_model


# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    model.eval()

    train_dataset, test_dataset = get_dataset()
    test_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=False, batch_size=args.test_batch_size)
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


def main():

    # load trained model
    model = get_model(args)
    if args.prune:
        if os.path.isfile(args.prune):
            print("=> loading checkpoint '{}'".format(args.prune))
            checkpoint = torch.load(args.prune)
            args.arch = checkpoint['arch']
            args.depth = checkpoint['depth']

            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model = get_model(best_acc)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}) Acc: {:f}"
                  .format(args.prune, checkpoint['epoch'], best_acc))
        else:
            print("=> no checkpoint found at '{}'".format(args.prune))


    if args.cuda:
        model.cuda()
    num_bn_channel, prune_threshold = get_bn_info(model)


    # pre prune
    pre_pruned_model, cfg, cfg_mask, pruned_ratio = pre_prune(model, prune_threshold, total=num_bn_channel)
    acc = test(pre_pruned_model)
    print(cfg)
    print('Pre-processing Successful!')

    # prune
    pruned_model = get_model(args, cfg=cfg)
    if args.cuda:
        pruned_model.cuda()
    print(pruned_model)
    num_parameters = sum([param.nelement() for param in pruned_model.parameters()])
    pruned_info = os.path.join(args.checkpoint, "prune.txt")
    with open(pruned_info, "w") as fp:
        fp.write("Configuration: \n"+str(cfg)+"\n")
        fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
        fp.write("Test accuracy: \n"+str(acc))

    pruned_model = execute_pruned(pre_pruned_model, pruned_model, cfg_mask)
    torch.save({'cfg': cfg, 'state_dict': pruned_model.state_dict()}, os.path.join(args.checkpoint, 'pruned.pth.tar'))

    pruned_acc = test(pruned_model)
    print(pruned_acc)


if __name__ == "__main__":
    main()