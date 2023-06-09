#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

#import matplotlib
#matplotlib.use('Agg')
import sys
import os
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import torch.nn as nn
import xlwt
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from tqdm import tqdm
from utils.sampling import mnist_iid, mnist_noniid, emnist_noniid, cifar_iid, cifar_noniid, select_adaptively, select_clustering, select_uniformly, get_matrix_similarity_from_grads
from utils.options import args_parser
from models.Update import LocalUpdate, DatasetSplit
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img, test_avg
from optimizers.fedoptimizer import FedMosOptimizer

if __name__ == '__main__':
    torch.set_printoptions(threshold=np.inf)
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)#return the dataset contained by each user
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
            # dict_users = torch.load('../data/mnist_noniid_dict.pt')
    elif args.dataset == 'emnist':
        trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.EMNIST('../data/emnist/', split='balanced', train=True, download=True, transform=trans_emnist)
        dataset_test = datasets.EMNIST('../data/emnist/', split='balanced', train=False, download=True, transform=trans_emnist)
        if args.iid:
            pass
        else:
            dict_users = emnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            # if os.path.exists('../data/cifar_noniid_dict_5labels.pt'):
            #     dict_users = torch.load('../data/cifar_noniid_dict_5labels.pt')
            # else:
            #     print('sampling ...')
            dict_users = cifar_noniid(dataset_train, args.num_users, args.alpha)
            
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'emnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    # net_glob.train()
    # copy weights
    w_glob = net_glob.state_dict()
    ut_glob = copy.deepcopy(w_glob)
    for k in ut_glob.keys():
        ut_glob[k] = torch.zeros_like(ut_glob[k]).to(args.device) #initialize the server momentum
    # ut_glob = torch.load('u_parameter_'+args.model+'_'+args.dataset+'.pt')
    
    '''
    generate the aggregation weight
    '''
    # p = np.random.uniform(1,2, args.num_users) #generate the weight, not 
    # p /= sum(p)
    # print(p)
    sample_num = np.array([len(dict_users[i]) for i in dict_users.keys()])
    p = sample_num / sum(sample_num)
    # while 1:
    #     p = np.random.dirichlet(np.ones(args.num_users))
    #     if max(p) - min(p) < 1/args.num_users: #do not want the weight difference to be too big
    #         break
        
    # training
    loss_train_glob = []
    acc_train_glob, acc_test_glob = [], []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.select == 'clustering':
        clients_grad = []
        net_pre = copy.deepcopy(net_glob)
        net_pre.train()
        # local_net = copy.deepcopy(net_glob)
        w_glob_pre = copy.deepcopy(w_glob)
        ut_glob_pre = copy.deepcopy(ut_glob)
        for iter_pre in range(args.epochs_pre):
            w_locals = []
            for idx in tqdm(dict_users.keys(), total=len(dict_users)):
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, local_grad = local.train(net=copy.deepcopy(net_pre).to(args.device))
                if iter_pre == args.epochs_pre-1:
                    clients_grad.append(local_grad)
            w_locals.append(copy.deepcopy(w))
            xt = FedAvg(w_locals,w_glob_pre,range(args.num_users),p) #get the avrage gradient
            for k in w_glob_pre.keys():
                w_glob_pre[k].add_(ut_glob_pre[k], alpha=(-args.lr*args.local_ep*args.beta))
                w_glob_pre[k].add_(xt[k])
            net_pre.load_state_dict(w_glob_pre)
        distance_type = 'cosine'
        num_clusters = 25
        sim_matrix = get_matrix_similarity_from_grads(clients_grad, distance_type=distance_type)
        sim_matrix = squareform(sim_matrix)
        linkage_matrix = linkage(sim_matrix, "ward")
        client_to_cluster = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')
        cluster_to_client = []
        for n in range(1, num_clusters+1):
            cluster_to_client.append(list(np.where(client_to_cluster == n)[0]))
        # print(cluster_to_client)
        cluster_weight = torch.zeros(num_clusters)
        for i in range(num_clusters):
            cluster_weight[i] += sum([p[j] for j in list(cluster_to_client[i])])
        cluster_to_client = sorted(cluster_to_client, key=lambda x:cluster_weight[cluster_to_client.index(x)], reverse=True)
        client_sorted = []
        for i in range(len(cluster_to_client)):
            cluster_to_client[i] = sorted(cluster_to_client[i], key=lambda x:p[x], reverse=True)
            client_sorted += cluster_to_client[i]


    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        net_glob.train()

        # loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        if args.select == 'uniform':
            idxs_users, wt =  select_uniformly(p,m,args.num_users)
        elif args.select == 'adaptive':
            idxs_users, wt =  select_adaptively(p,m,args.num_users)
        elif args.select == 'clustering':
            idxs_users, wt = select_clustering(client_sorted,p,m,args.num_users)

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, _ = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            # loss_locals.append(copy.deepcopy(loss))
        # update global weights
        xt = FedAvg(w_locals,w_glob,idxs_users,wt) #get the avrage gradient
        if iter == 0: #no server momentum update
            for k in w_glob.keys():
                # w_glob[k].add_(1/args.local_ep, xt[k])
                w_glob[k].add_(ut_glob[k], alpha=(-args.lr*args.local_ep*args.beta))
                w_glob[k].add_(xt[k])

        else:
            for k in ut_glob.keys():
                '''
                server momentum
                '''
                ut_glob[k] = ut_glob[k]*args.beta - xt[k]/(args.lr*args.local_ep)
                w_glob[k] -= ut_glob[k]*args.local_ep*args.lr

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # calculate loss
        loss_locals = []
        with torch.no_grad():
            criterion = nn.CrossEntropyLoss()
            # loss_locals = []
            for idx in dict_users:
                dl_train = DataLoader(DatasetSplit(dataset_train, dict_users[idx]), batch_size=args.local_bs, shuffle=True)
                loss = 0
                for batch_idx, (images, labels) in enumerate(dl_train):
                    images, labels = images.to(args.device), labels.to(args.device)
                    log_probs = net_glob(images)
                    loss += criterion(log_probs, labels)
                loss /= batch_idx+1
                loss_locals.append(loss.item())
        loss_avg = sum(np.asarray(loss_locals)*p)

        # print loss
        #loss_avg = sum(loss_locals) / len(loss_locals)
        # loss_avg = sum(np.asarray(loss_locals)*wt[idxs_users]) #the loss is the weighted sum
        print('Round {:3d}, Average loss {:.3f}'.format(iter+1, loss_avg.item()))
        loss_train_glob.append(loss_avg)

        # testing
        # acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        # acc_test, loss_test = test_avg(net_glob, dataset_test, dict_users_test, p, args)

        # acc_train_glob.append(acc_train)
        acc_test_glob.append(acc_test)
        # print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))

    # results
    # torch.save(ut_glob, 'u_parameter2_'+args.model+'_'+args.dataset+'.pt')
    workbook = xlwt.Workbook()
    worksheet1 = workbook.add_sheet('loss')
    # worksheet2 = workbook.add_sheet('acc_train')
    worksheet2 = workbook.add_sheet('acc_test')
    for i in range(len(loss_train_glob)):
        worksheet1.write(i,0,float(loss_train_glob[i]))
    # for i in range(len(acc_train_glob)):
    #     worksheet2.write(i,0,float(acc_train_glob[i]))
    for i in range(len(acc_test_glob)):
        worksheet2.write(i,0,float(acc_test_glob[i]))
    if args.select == 'clustering':
        workbook.save('./save/{}/fed_{}_{}_{}_iid{}_{}_B{}_E{}_lr{}_mu{}_a{}_beta{}_t{}_Te{}_seed{}.xls'.format(
            args.dataset, args.dataset, args.model, args.frac, args.iid, args.select,
            args.local_bs, args.local_ep, args.lr, args.mu, args.a, args.beta, num_clusters, args.epochs_pre, args.seed))
    else:
        workbook.save('./save/{}/fed_{}_{}_{}_iid{}_{}_B{}_E{}_lr{}_mu{}_a{}_beta{}_seed{}.xls'.format(
        args.dataset, args.dataset, args.model, args.frac, args.iid, args.select,
        args.local_bs, args.local_ep, args.lr, args.mu, args.a, args.beta, args.seed))
