#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w,w1,idxs_users,wt): #rewrite to calculate the gradient instead of simple average
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = (w_avg[k]-w1[k])*wt[idxs_users[0]]
        for i in range(1, len(w)):
            w_avg[k] += (w[i][k]-w1[k])*wt[idxs_users[i]]
        #w_avg[k] = torch.div(w_avg[k], len(w)) #no longer simple averaging
    return w_avg
