#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tools.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/5 下午5:34
# @ Software   : PyCharm
#------------------------------------------------------

import os
import torch
import shutil

import models
from configs.cfgs import args


def get_model(args, cfg=None):

    if cfg is not None:
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=cfg)
    else:
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

    return model


def save_checkpoint(state, is_best, filepath):
    """

    :param state:
    :param is_best:
    :param filepath:
    :return:
    """
    os.makedirs(filepath, exist_ok=True)
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

if __name__ == "__main__":

    model = get_model(args)

    print(model)