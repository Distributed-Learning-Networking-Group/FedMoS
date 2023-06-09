#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from models.Update import DatasetSplit


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        if args.emnist_letters:
            target -= 1
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

@torch.no_grad()
def test_avg(net_g, dataset, dict_users, weight, args):
    net_g.eval()
    test_loss = []
    test_accuracy = []
    for user_idx in dict_users:
        data_loader = DataLoader(DatasetSplit(dataset, dict_users[user_idx]), batch_size=args.bs, shuffle=False)
        # print('user idx:', user_idx)
        # print(len(data_loader.dataset))
        loss, correct = 0, 0
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            # print(correct)

        loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        test_loss.append(loss)
        test_accuracy.append(accuracy)

    weighted_loss = sum(np.asarray(test_loss)*weight)
    weighted_accuracy = sum(np.asarray(test_accuracy)*weight)
    
    return weighted_accuracy, weighted_loss