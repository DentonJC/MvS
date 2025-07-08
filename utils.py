#!/usr/bin/env python
# coding: utf-8

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.initial_seed()
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
