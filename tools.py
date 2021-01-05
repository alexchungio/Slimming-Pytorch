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

import torch
import models
from configs.cfgs import args


def get_model(args, cfg=None):

    if cfg is not None:
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=cfg)
    else:
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

    return model


if __name__ == "__main__":

    model = get_model(args)

    print(model)