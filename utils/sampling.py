#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import numpy as np
import copy
from torchvision import datasets, transforms
from numpy.random import dirichlet
from itertools import product

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users #evenly divide all the tranining data to all users

def mnist_noniid(dataset, num_users):
    # np.random.seed(29)
    num_dataset = len(dataset)
    idxs = np.arange(num_dataset)
    labels = dataset.train_labels.numpy()
    dict_users = {i: list() for i in range(num_users)}
    if num_users == 100:
        min_num, max_num = 200, 1000
    elif num_users == 500:
        # min_num, max_num = 40, 200 # 13
        min_num, max_num = 10, 50 # 20
    random_num_size = np.random.randint(min_num, max_num+1, size=num_users)
    # assert sum(random_num_size) <= num_dataset

    # sort by labels
    mnist_data = []
    for label in range(10):
        i = labels==label
        mnist_data.append(idxs[i])
    # label_times = np.zeros(10)
    # data_times = np.zeros(10)
    # divide and assign
    for i, num in enumerate(random_num_size):
        rand_num = []
        rand_num.append(np.random.randint(int(min_num/4), num))
        rand_num.append(num - rand_num[0])
        total_num = sum([len(j) for j in mnist_data])
        p = [len(j)/total_num for j in mnist_data]
        labels_assign = np.random.choice(range(10), 2, p=p, replace=False)
        for label, num in zip(labels_assign, rand_num):
            if(num >= len(mnist_data[label])):
                num = len(mnist_data[label])
            # label_times[label] += 1
            # data_times[label] += num
            rand_set = set(np.random.choice(mnist_data[label], num, replace=False))
            mnist_data[label]=list(set(mnist_data[label])-rand_set)
            dict_users[i] += rand_set
    # print(label_times)
    # print(data_times)
    print(f"Total number of datasets owned by clients : {sum([len(dict_users[i]) for i in dict_users.keys()])}")
    return dict_users

def emnist_noniid(dataset, num_users):
    # np.random.seed(29)
    num_dataset = len(dataset)
    idxs = np.arange(num_dataset)
    labels = dataset.targets.numpy()
    dict_users = {i: list() for i in range(num_users)}
    if num_users == 500:
        min_num, max_num = 50, 400 #16/17 balanced
        # min_num, max_num = 80, 400 #18 balanced
    # elif num_users == 1000:
    #     min_num, max_num = 200, 1000
    random_num_size = np.random.randint(min_num, max_num+1, size=num_users)
    num_labels = 10
    num_total_label = labels.max()+1

    # sort by labels
    mnist_data = []
    for label in range(num_total_label):
        i = labels==label
        mnist_data.append(idxs[i])
    # label_times = np.zeros(num_total_label)
    # data_times = np.zeros(num_total_label)

    # divide and assign
    total_num = sum([len(j) for j in mnist_data])
    p = [len(j)/total_num for j in mnist_data]
    for i, n in enumerate(random_num_size):
        num_1_shard = int(n/num_labels)
        total_num = sum([len(j) for j in mnist_data])
        p = [len(j)/total_num for j in mnist_data]
        labels_assign = np.random.choice(range(num_total_label), num_labels, p=p, replace=False)
        # print('================== {} ================='.format(i))
        # print(labels_assign)
        for label in labels_assign:
            num = num_1_shard
            if(num >= len(mnist_data[label])):
                num = len(mnist_data[label])
            # label_times[label] += 1
            # data_times[label] += num
            rand_set = set(np.random.choice(mnist_data[label], num, replace=False))
            mnist_data[label]=list(set(mnist_data[label])-rand_set)
            dict_users[i] += rand_set
    # print(label_times)
    # print(data_times)
    print(f"Total number of datasets owned by clients : {sum([len(dict_users[i]) for i in dict_users.keys()])}")
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

"""5 shards"""
def cifar_noniid(dataset, num_users, alpha=0.1):
    # np.random.seed(42)
    num_dataset = len(dataset)
    idxs = np.arange(num_dataset)
    labels = np.array(dataset.targets)
    dict_users = {i: list() for i in range(num_users)}
    if num_users == 100:
        num_samples = [1000] * 10 + [750] * 20 + [500] * 30 + [250] * 30 + [100] * 10
    elif num_users == 500:
        num_samples = [200] * 50 + [150] * 100 + [100] * 150 + [50] * 150 + [20] * 50
    num_labels = 5

    # sort by labels
    mnist_data = []
    for label in range(10):
        i = labels==label
        mnist_data.append(idxs[i])

    label_times = np.zeros(10)
    data_times = np.zeros(10)

    # divide and assign
    for i, n in enumerate(num_samples):
        num_1_shard = int(n/num_labels)
        total_num = sum([len(j) for j in mnist_data])
        p = [len(j)/total_num for j in mnist_data]
        labels_assign = np.random.choice(range(10), num_labels, p=p, replace=False)
        # print('================== {} ================='.format(i))
        # print(labels_assign)
        for label in labels_assign:
            num = num_1_shard
            if(num >= len(mnist_data[label])):
                num = len(mnist_data[label])
            label_times[label] += 1
            data_times[label] += num
            rand_set = set(np.random.choice(mnist_data[label], num, replace=False))
            mnist_data[label]=list(set(mnist_data[label])-rand_set)
            dict_users[i] += rand_set
    # print(label_times)
    # print(data_times)
    print(f"Total number of datasets owned by clients : {sum([len(dict_users[i]) for i in dict_users.keys()])}")
    return dict_users


"""2 shards"""
# def cifar_noniid(dataset, num_users, alpha=0.1):
#     # np.random.seed(42)
#     num_dataset = len(dataset)
#     idxs = np.arange(num_dataset)
#     labels = np.array(dataset.targets)
#     dict_users = {i: list() for i in range(num_users)}
#     min_num = 200
#     max_num = 800
#     random_num_size = np.random.randint(min_num, max_num+1, size=num_users)
#     # assert sum(random_num_size) <= num_dataset

#     # sort by labels
#     mnist_data = []
#     for label in range(10):
#         i = labels==label
#         mnist_data.append(idxs[i])

#     # label_times = np.zeros(10)
#     # data_times = np.zeros(10)

#     # divide and assign
#     for i, n in enumerate(random_num_size):
#         rand_num = []
#         rand_num.append(np.random.randint(1, n))
#         rand_num.append(n - rand_num[0])
#         total_num = sum([len(j) for j in mnist_data])
#         p = [len(j)/total_num for j in mnist_data]
#         labels_assign = np.random.choice(range(10), 2, p=p, replace=False)
#         # print('================== {} ================='.format(i))
#         # print(labels_assign)
#         for label, num in zip(labels_assign, rand_num):
#             if(num >= len(mnist_data[label])):
#                 num = len(mnist_data[label])
#             # print(i, label, num)
#             # label_times[label] += 1
#             # data_times[label] += num
#             rand_set = set(np.random.choice(mnist_data[label], num, replace=False))
#             mnist_data[label]=list(set(mnist_data[label])-rand_set)
#             dict_users[i] += rand_set
#     # print(label_times)
#     # print(data_times)
#     print(f"Total number of datasets owned by clients : {sum([len(dict_users[i]) for i in dict_users.keys()])}")
#     return dict_users

"""dirichlet distribution"""
# def cifar_noniid(dataset, num_users, alpha=1.0):
#     num_dataset = len(dataset)
#     idxs_train = np.arange(num_dataset)
#     train_labels = np.array(dataset.targets)

#     # the number of train and test samples assigned to every client
#     # num_samples = [100] * 10 + [250] * 30 + [500] * 30 + [750] * 20 + [1000] * 10
#     num_samples = [1000]*1 + [500]*1 + [400]*1 + [300]*1 + [200]*50 + [150]*100 + [100]*150 + [50]*100 + [20]*96

#     dict_users = {i: list() for i in range(num_users)}

#     matrix = dirichlet([alpha] * 10, size=num_users)

#     # sort by labels
#     mnist_data = []
#     for label in range(10):
#         i = train_labels==label
#         mnist_data.append(idxs_train[i])

#     # label_times = np.zeros(10)
#     # data_times = np.zeros(10)
#     all_data = []

#     # divide and assign training data
#     for user_idx, num_sample in enumerate(num_samples):
#         user_samples = 0
#         for label in range(10):
#             if label < 9:
#                 num_label = int(matrix[user_idx, label] * num_sample)
#             else:
#                 num_label = num_sample - user_samples
#             if num_label > len(mnist_data[label]):
#                 num_label = len(mnist_data[label])
#             user_samples += num_label
#             # label_times[label] += 1
#             # data_times[label] += num_label
#             rand_set = set(np.random.choice(mnist_data[label], size=num_label, replace=False))
#             mnist_data[label] = list(set(mnist_data[label]) - rand_set)
#             dict_users[user_idx] += rand_set
#             all_data += rand_set

#     # print(label_times)
#     # print(data_times)
#     print(f"Total number of training datasets owned by clients : {([len(dict_users[i]) for i in dict_users.keys()])}")
#     # print(f"Remove duplicate data : {len(list(set(all_data)))}")
#     return dict_users

def select_adaptively(p,m,num_users):
    P_acu = np.zeros(num_users) #record the accumated probabaility
    p_user = np.zeros((num_users,2))
    for i in range(num_users):
        p_user[i,0] = p[i]
        p_user[i,1] = i
    p_user = p_user[(-p_user[:,0]).argsort()] #sort the users according to the first weight
    # print(p_user)
    '''
    the main selection process
    '''
    idxs_users = []
    wt = np.zeros(num_users) #the client weight
    for k in range(m):
        psum = 0
        pk = np.zeros(num_users)
        for i in range(num_users):
            if P_acu[i] < m*p_user[i,0]:
                pk[i] = min(m*p_user[i,0]-P_acu[i],1)
                psum1 = psum
                psum += pk[i]
                if psum <= 1:
                    P_acu[i] += pk[i]
                else:
                    pk[i] = 1-psum1
                    P_acu[i] += pk[i]
                    break
        # print(pk)
        idx = np.random.choice(range(num_users), 1, p=pk)
        wt[int(p_user[idx[0],1])] += 1
        if int(p_user[idx[0],1]) in idxs_users:
            continue
        idxs_users.append(int(p_user[idx[0],1])) #add the client to the list
    wt /= m
    # print(idxs_users)
    return idxs_users, wt
    
def select_uniformly(p,m,num_users):
    idxs_users = np.random.choice(range(num_users), m, replace=False, p=p)
    wt = np.zeros(num_users)
    for idx in idxs_users:
        wt[idx] = num_users*p[idx]/m
    wt /= sum(wt)
    return idxs_users, wt

def get_similarity(grad_1, grad_2, distance_type="L1"):
    if distance_type == "L1":
        norm = 0
        for g_1, g_2 in zip(grad_1, grad_2):
            norm += np.sum(np.abs(g_1 - g_2))
        return norm

    elif distance_type == "L2":
        norm = 0
        for g_1, g_2 in zip(grad_1, grad_2):
            norm += np.sum((g_1 - g_2) ** 2)
        return norm

    elif distance_type == "cosine":
        norm, norm_1, norm_2 = 0, 0, 0
        for i in range(len(grad_1)):
            norm += np.sum(grad_1[i] * grad_2[i])
            norm_1 += np.sum(grad_1[i] ** 2)
            norm_2 += np.sum(grad_2[i] ** 2)

        if norm_1 == 0.0 or norm_2 == 0.0:
            return 0.0
        else:
            norm /= np.sqrt(norm_1 * norm_2)

            return np.arccos(norm)

def get_matrix_similarity_from_grads(local_model_grads, distance_type='cosine'):
    """return the similarity matrix where the distance chosen to
    compare two clients is set with `distance_type`"""

    num_clients = len(local_model_grads)
    metric_matrix = np.zeros((num_clients, num_clients))

    for i, j in product(range(num_clients), range(num_clients)):
        metric_matrix[i, j] = get_similarity(
            local_model_grads[i], local_model_grads[j], distance_type
        )
    return metric_matrix


def select_clustering(client_sorted, p, m, num_users):
    P_acu = np.zeros(num_users) #record the accumated probabaility
    p_user = np.zeros((num_users,2))
    for i, user in enumerate(client_sorted):
        p_user[i,0] = p[user]
        p_user[i,1] = user
    # p_user = p_user[(-p_user[:,0]).argsort()] #sort the users according to the first weight
    #print(p_user)
    '''
    the main selection process
    '''
    idxs_users = []
    wt = np.zeros(num_users) #the client weight
    for k in range(m):
        psum = 0
        pk = np.zeros(num_users)
        for i in range(num_users):
            if P_acu[i] < m*p_user[i,0]:
                pk[i] = min(m*p_user[i,0]-P_acu[i],1)
                psum1 = psum
                psum += pk[i]
                if psum <= 1:
                    P_acu[i] += pk[i]
                else:
                    pk[i] = 1-psum1
                    P_acu[i] += pk[i]
                    break
        # print(pk)
        idx = np.random.choice(range(num_users), 1, p=pk)
        wt[int(p_user[idx[0],1])] += 1
        if int(p_user[idx[0],1]) in idxs_users:
            continue
        idxs_users.append(int(p_user[idx[0],1])) #add the client to the list
    wt /= m
    # print(idxs_users)
    # print(wt)
    return idxs_users, wt

if __name__ == '__main__':
    # dataset_train = datasets.MNIST('code/data/mnist/', train=True, download=True,
    #                                 transform=transforms.Compose([
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize((0.1307,), (0.3081,))
    #                                 ]))
        
    # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # dataset_train = datasets.CIFAR10('code/data/cifar', train=True, download=True, transform=trans_cifar)
    
    trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.EMNIST('../../data/emnist/', split='letters', train=True, download=True, transform=trans_emnist)
    dataset_test = datasets.EMNIST('../../data/emnist/', split='letters', train=False, download=True, transform=trans_emnist)
    print(len(dataset_train.targets))
    print(sorted((dataset_test.targets.numpy())))

    num_users = 500
    m = int(num_users*0.1)
    np.random.seed(29)
    d = emnist_noniid(dataset_train, num_users)
    sample_num = np.array([len(d[i]) for i in d.keys()])
    p = sample_num / sum(sample_num)

    # idxs_users, wt = select_adaptively(p,m,num_users)
    #     print(wt)

    # p = np.random.dirichlet(np.ones(num_users))
    # idxs_users, wt = select_adaptively(p, m, num_users)
