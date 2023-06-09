#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy
from optimizers.fedoptimizer import FedMosOptimizer

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.lda_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=len(idxs), shuffle=True)#load all data

    # def train(self, net):
    #     net.train()
    #     net_prev = copy.deepcopy(net).to(self.args.device)
    #     local_net = copy.deepcopy(net).to(self.args.device) # first copy the initial network for prox
    #     # train and update
    #     # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
    #     optimizer = FedMosOptimizer(net.parameters(), lr=self.args.lr, a=self.args.a, mu=self.args.mu)
    #     # epoch_loss = []
    #     # netb = copy.deepcopy(net).to(self.args.device) # record the parameter before updating
    #     ldt = [torch.zeros_like(p.data) for p in net.parameters() if p.requires_grad]# local momentum
    #     # grad_prev = [torch.zeros_like(p.data) for p in net.parameters() if p.requires_grad]
    #     Lldt =len(ldt)
    #     for iter in range(self.args.local_ep):
    #         for batch_idx, (images, labels) in enumerate(self.ldr_train):
    #             images, labels = images.to(self.args.device), labels.to(self.args.device)
    #             net.zero_grad()
    #             net_prev.zero_grad()
    #             log_probs, log_probs_prev = net(images), net_prev(images)
    #             loss, loss_prev = self.loss_func(log_probs, labels), self.loss_func(log_probs_prev, labels)
    #             loss.backward()
    #             loss_prev.backward()

    #             optimizer.update_momentum(net_prev)
    #             optimizer.step(copy.deepcopy(local_net))
    #             net_prev = copy.deepcopy(net)
    #             # optimizer.clone_grad()
    #         #     batch_loss.append(loss.item())
    #         # epoch_loss.append(sum(batch_loss)/len(batch_loss))

    #     return net.state_dict(), optimizer.get_grad()

    def train(self, net):
        net.train()
        local_net = copy.deepcopy(net).to(self.args.device) # first copy the initial network for prox
        # train and update
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer = FedMosOptimizer(net.parameters(), lr=self.args.lr, a=self.args.a, mu=self.args.mu)
        # epoch_loss = []
        # netb = copy.deepcopy(net).to(self.args.device) # record the parameter before updating
        ldt = [torch.zeros_like(p.data) for p in net.parameters() if p.requires_grad]# local momentum
        # grad_prev = [torch.zeros_like(p.data) for p in net.parameters() if p.requires_grad]
        Lldt =len(ldt)
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                
                optimizer.update_momentum()
                optimizer.step(copy.deepcopy(local_net))
                # optimizer.clone_grad()
            #     batch_loss.append(loss.item())
            # epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), optimizer.get_grad()
        

        for iter in range(self.args.local_ep):
            batch_loss = []
            if iter == 0:
                for batch_idx, (images, labels) in enumerate(self.lda_train):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    batch_loss.append(loss.item())
                    if self.args.verbose and batch_idx % 10 ==0:
                        print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            iter, batch_idx * len(images), len(self.ldr_train.dataset),
                            100. * batch_idx / len(self.ldr_train), loss.item()))
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
                    
                '''
                initilize the local momentum
                '''
                for p, i in zip(net.parameters(),range(Lldt)):
                    if p.grad is None:
                        continue
                    ldt[i] = p.grad.data.clone()
                    grad_prev[i] = p.grad.data.clone()

                for group in optimizer.param_groups:
                    for p, i, x, lx in zip(group['params'], range(Lldt), net.parameters(), local_net.parameters()):
                        p.data.add_(ldt[i], alpha=(-self.args.lr))#here is the parameter update
                        p.data.add_(x.data-lx.data, alpha=(-self.args.mu))

            else:
                tr_data = self.ldr_train#current sampled data             
                for batch_idx, (images, labels) in enumerate(tr_data):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    # netb.zero_grad(), net.zero_grad()
                    # logb_probs, log_probs = netb(images), net(images)
                    # lossb, loss =self.loss_func(logb_probs, labels), self.loss_func(log_probs, labels)
                    # lossb.backward(), loss.backward()
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    # optimizer.step()
                    batch_loss.append(loss.item())
                    if self.args.verbose and batch_idx % 10 ==0:
                        print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            iter, batch_idx * len(images), len(self.ldr_train.dataset),
                            100. * batch_idx / len(self.ldr_train), loss.item()))
                # epoch_loss.append(sum(batch_loss)/len(batch_loss))
                    '''
                    update local momentum
                    '''
                    # for pb, p, i in zip(netb.parameters(), net.parameters(), range(Lldt)):
                    #     ldt[i] = p.grad.data +(1-self.args.a)*(ldt[i] - pb.grad.data)
                    for p, i in zip(net.parameters(), range(Lldt)):
                        ldt[i] = p.grad.data +(1-self.args.a)*(ldt[i] - grad_prev[i])
                
                    for i, p in enumerate(net.parameters()):
                        grad_prev[i] = p.grad.data.clone()
                        

                    '''
                    update the parameter, rewrite the step
                    '''
                    # netb = copy.deepcopy(net).to(self.args.device)#clone the original net
                    for group in optimizer.param_groups:
                        for p, i, x, lx in zip(group['params'], range(Lldt), net.parameters(), local_net.parameters()):
                            p.data.add_(ldt[i], alpha=(-self.args.lr))#here is the parameter update
                            p.data.add_(x.data-lx.data, alpha=(-self.args.mu))
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)