import os

# from torchvision import datasets, transforms

import argparse
import time
import random
import time
import math
import numpy as np
from utils import dataset_split

import torch
import torch.nn as nn
import torch.optim as optim
from deeplearning import FederatedLearning, Selected_DataSet,MyDataSet
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import fixed_seed, check_args
from runx.logx import logx
import pandas as pd

torch.multiprocessing.set_sharing_strategy('file_system')
from utils import load_model, load_dataset
from tqdm import tqdm
import pickle
from scipy.ndimage import median_filter
import cv2
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from time_logger import TimeLogger
import seaborn as sns

w1, h1 = 224, 224
w2, h2 = 28, 28


def fl_run(args):
    logx.initialize(logdir=args.logdir, coolname=False, tensorboard=False, hparams=args)
    logx.msg(f'====================== Federated Learning Running ====================')
    FL = FederatedLearning(args)
    time_loger = TimeLogger(args.logdir+'/fl_train.csv')
    FL.run()
    time_loger.log_time('fl_train_finish')


def AR_search_interplation(args):
    p_trainset, p_testset, _ = load_dataset(args, args.p_dataset)
    h_trainset, h_testset, _ = load_dataset(args, args.h_dataset)
    h_num_classes = args.h_num_classes
    p_num_classes = args.p_num_classes
    model = load_model(args.model, p_num_classes)
    if args.p_dataset == 'tinyimagenet100':
        h_dataset = 'gtsrb'
    elif args.p_dataset == 'gtsrb':
        h_dataset = 'cifar10'
    elif args.p_dataset == 'cifar10':
        h_dataset = 'mnist'
    elif args.p_dataset == 'svhn':
        h_dataset = 'cifar10'
    model_path = f'results/global_model/{args.model}/{args.p_dataset}'
    # save_path = f'results/{args.date}/{args.train_mode}/{args.model}-{args.p_dataset}-{args.h_dataset}-{args.eta}-{args.normal_local_epochs}-{args.attacker_local_epochs}-{args.number_clients_total}-{args.number_clients_of_round}-{args.number_attackers_of_round}-{args.AR_attack_rounds}-{args.alpha}'
    # f'{args.logdir}/{args.date}/{args.train_mode}/{args.model}-{args.p_dataset}-{args.h_dataset}-{args.eta}-{args.normal_local_epochs}-{args.attacker_local_epochs}-{args.number_clients_total}-{args.number_clients_of_round}-{args.number_attackers_of_round}-{args.AR_attack_rounds}-{args.alpha}'
    model_path += f'/{args.AR_attack_rounds - 1}.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.to(args.device)
    model.eval()
    model.get_features()

    # Training sort images and labels by class 
    train_sorted_images = [[] for _ in range(h_num_classes)]  # 10 classes in MNIST
    train_sorted_labels = [[] for _ in range(h_num_classes)]

    for image, label in h_trainset:
        train_sorted_images[label].append(image)
        train_sorted_labels[label].append(label)

    # Testing sort images and labels by class 
    test_sorted_images = [[] for _ in range(h_num_classes)]  # 10 classes in MNIST
    test_sorted_labels = [[] for _ in range(h_num_classes)]

    for image, label in h_testset:
        test_sorted_images[label].append(image)
        test_sorted_labels[label].append(label)

    train_sorted_dataset = []
    manual_mapping_dict = {
    8:20
    }

    for class_index in range(h_num_classes):
        original_label_list = train_sorted_labels[class_index]
        original_label = original_label_list[0]
        
        target_label = manual_mapping_dict.get(original_label, original_label)
        num_images = len(train_sorted_images[class_index])
        new_labels_list = [target_label] * num_images
        train_sorted_dataset.extend(list(zip(train_sorted_images[class_index], new_labels_list)))

    train_sorted_loader = torch.utils.data.DataLoader(train_sorted_dataset, batch_size=args.batchsize, shuffle=True)

    W = torch.randn((3, args.img_size, args.img_size)).to(args.device)
    W = W.detach()  # 固定梯度
    W.requires_grad = True

    BCE = torch.nn.BCELoss()
    optimizer = torch.optim.Adam([W], lr=0.005, betas=(0.5, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)

    # Example usage of sorted_loader:
    for i in range(30):
        for j, (images, labels) in enumerate(train_sorted_loader):
            batch_size = images.shape[0]
            if images.shape[1] == 1:
                images = np.tile(images, (1, 3, 1, 1))  # 扩充通道数
                X = torch.from_numpy(images).to(args.device)
            else:
                X = images.to(args.device)
            labels = torch.zeros(batch_size, h_num_classes).scatter_(1, labels.view(-1, 1), 1)
            labels = labels.to(args.device)

            # X = np.zeros((batch_size, 3, h1, w1), dtype=np.float32)
            # X = images
            # X = torch.from_numpy(X).to(args.device)

            P = torch.sigmoid(W)
            X_adv = args.alpha * X + (1 - args.alpha) * P  # range [0, 1]

            Y_adv = model(X_adv)
            Y_adv = F.softmax(Y_adv, 1)
            out = Y_adv[:, :h_num_classes]
            loss = BCE(out, labels)  # + args.ar_lmd * torch.norm(W) ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if (j + 1) % 100 == 0:
                print('epoch %03d/%03d, batch %06d, loss %.6f' % (i + 1, 30, j + 1, loss.data.cpu().numpy()))
        avg_acc = 0
        for _class_index in range(h_num_classes):
            test_sorted_dataset = []
            test_sorted_dataset.extend(list(zip(test_sorted_images[_class_index], test_sorted_labels[_class_index])))
            test_sorted_loader = torch.utils.data.DataLoader(test_sorted_dataset, batch_size=args.batchsize,
                                                             shuffle=False)
            acc = 0
            for images, labels in test_sorted_loader:
                if images.shape[1] == 1:
                    images = np.tile(images, (1, 3, 1, 1))
                    X = torch.from_numpy(images).to(args.device)
                else:
                    X = images.to(args.device)
                # label = torch.zeros(batch_size, 10).scatter_(1, label.view(-1,1), 1)
                # label = tensor2var(label)
                # labels = labels * 0 + class_index

                # X = torch.zeros(batch_size, 3, h1, w1)
                # X[:,:,(h1-h2)//2:(h1+h2)//2, (w1-w2)//2:(w1+w2)//2] = torch.from_numpy(images)
                # X = X.to(args.device)
                # P = torch.sigmoid(W * M)
                # X_adv = X + P # range [0, 1]

                P = torch.sigmoid(W)
                X_adv = args.alpha * X + (1 - args.alpha) * P  # range [0, 1]

                Y_adv = model(X_adv)
                Y_adv = F.softmax(Y_adv, 1)
                out = Y_adv[:, :h_num_classes]
                pred = out.data.cpu().numpy().argmax(1)
                acc += sum(labels.numpy() == pred) / float(len(labels) * len(test_sorted_loader))
            print('class %02d, epoch %03d/%03d, test accuracy %.6f' % (_class_index, i, 30, acc))
            avg_acc += acc
        avg_acc /= h_num_classes
        print('epoch %03d/%03d, avg accuracy %.6f' % (i, 30, avg_acc))
        # print(W)
    ##############################


def extract_feature_base_original(model, p_trainset, p_num_classes):
    # Training sort images and labels by class
    train_sorted_images = [[] for _ in range(p_num_classes)]  # 10 classes in MNIST
    train_sorted_labels = [[] for _ in range(p_num_classes)]

    for image, label in p_trainset:
        if label < p_num_classes:
            train_sorted_images[label].append(image)
            train_sorted_labels[label].append(label)

    Features = []
    for class_index in range(p_num_classes):
        train_sorted_dataset = []
        train_sorted_dataset.extend(list(zip(train_sorted_images[class_index], train_sorted_labels[class_index])))
        train_sorted_loader = torch.utils.data.DataLoader(train_sorted_dataset, batch_size=args.batchsize, shuffle=True)
        # all_features = []
        for j, (images, labels) in enumerate(train_sorted_loader):
            images = images.to(args.device)
            outputs = model.get_features(images)

            # all_features.append(outputs)
            features = outputs if j == 0 else torch.cat((features, outputs), dim=0)

        # 计算所有输出的平均值
        average_feature = torch.mean(features, dim=0).unsqueeze(0)

        Features.append(average_feature)
    return Features


def extract_feature_maxtop1(model, p_num_classes, h_num_classes, cor_map, neg_label):
    # class_list = list(range(h_num_classes))
    # class_list.append(p_num_classes-1)
    class_list = []
    for class_index in range(h_num_classes):
        cor_index = cor_map[class_index]
        class_list.append(cor_index)
    # neg_range = [i for i in range(p_num_classes) if i not in class_list]
    # class_list.append(random.choice(neg_range))
    class_list.append(neg_label)

    Features = []
    for class_index in class_list:

        # train_sorted_dataset = []
        # train_sorted_dataset.extend(list(zip(train_sorted_images[class_index], train_sorted_labels[class_index])))
        # train_sorted_loader = torch.utils.data.DataLoader(train_sorted_dataset, batch_size=args.batchsize, shuffle=True)

        labels = torch.tensor([class_index]).to(torch.int64)
        labels = torch.zeros(1, p_num_classes).scatter_(1, labels.view(-1, 1), 1)
        labels = labels.to(args.device)
        # print(labels)
        # print(type(labels))
        # exit()

        Fake_input = torch.randn((1, 3, args.img_size, args.img_size)).to(args.device)
        Fake_input = Fake_input.detach()
        Fake_input.requires_grad = True

        BCE = torch.nn.BCELoss()
        optimizer = torch.optim.Adam([Fake_input], lr=0.005, betas=(0.5, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)

        # Example usage of sorted_loader:
        for i in range(500):

            P = torch.sigmoid(Fake_input)

            Y = model(P)

            Y = F.softmax(Y, 1)
            # out = Y[:,:p_num_classes]
            loss = BCE(Y, labels)  # + args.ar_lmd * torch.norm(W) ** 2

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            lr_scheduler.step()
            # if (i +1) % 100 == 0:
            #     logx.msg('epoch %03d/%03d, loss %.6f' % (i + 1, 500, loss.data.cpu().numpy()))

        P = torch.sigmoid(Fake_input)
        feature = model.get_features(P)
        Features.append(feature)

    return Features


def predict_labels(args, model, train_loader, p_num_classes):
    """
    return:一个向量 (1*100) 表示一类h样本被预测到p的标签
    """
    model.eval()
    predict_label = []
    result = [0] * p_num_classes
    for i, (images, _, _) in enumerate(tqdm(train_loader)):
        if images.shape[1] == 1:
            images = np.tile(images, (1, 3, 1, 1))
            images = torch.from_numpy(images).to(args.device)
        images = images.to(args.device)
        y = model(images)
        y = y.max(1, keepdim=True)[1]
        y = y.detach().cpu().numpy()
        predict_label.extend(y)
    unique_value, counts = np.unique(predict_label, return_counts=True)
    for i in range(len(unique_value)):
        result[unique_value[i]] = counts[i]
    return result


def search_labels(args, matrix):
    """
    返回一个map key是h类 v是对应的p类
    该算法是从矩阵中最大值开始搜索
    """
    used_i = []
    used_j = []
    result_map = {}
    frequnecy_map = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            frequnecy_map[(i, j)] = matrix[i][j]
    sorted_map = dict(sorted(frequnecy_map.items(), key=lambda x: x[1], reverse=True))
    for (i, j) in sorted_map.keys():
        if i not in used_i and j not in used_j:
            result_map[i] = j
            used_i.append(i)
            used_j.append(j)
    column_sum = np.sum(matrix, axis=0)
    neg_labels = {}
    for i in range(args.p_num_classes):
        if i not in used_j:
            neg_labels[i] = column_sum[i]
    neg_label = max(neg_labels, key=neg_labels.get)
    return result_map, neg_label


def search_best_correspondence(args, model, train_sorted_images, p_num_classes, h_num_classes):
    """
    return:map,key:h_data_class value:p_num_class
    """
    pre_matrix = []  # 预测出来的所有标签，送入贪心算法,shape: 10*100
    for class_index in range(h_num_classes):
        train_sorted_dataset = Selected_DataSet(train_sorted_images, class_index)
        train_sorted_loader = torch.utils.data.DataLoader(train_sorted_dataset, batch_size=args.batchsize, shuffle=True)
        pre_matrix.append(predict_labels(args, model, train_sorted_loader, p_num_classes))
    pre_matrix = np.array(pre_matrix)
    result_map, neg_label = search_labels(args, pre_matrix)
    return result_map, neg_label



def AR_search_interplation_feature(args):
    logx.initialize(logdir=args.logdir, coolname=False, tensorboard=False, hparams=args)
    logx.msg(f'====================== AR_search_interplation_feature Running ====================')
    p_trainset, p_testset, p_data_classes = load_dataset(args, args.p_dataset, mode='h')
    p_num_classes = args.p_num_classes
    # client_dataset_size = int(len(p_trainset)/args.number_clients_total)
    # p_dataset_list = dataset_split(p_trainset, args.number_clients_total * [client_dataset_size])
    h_trainset, h_testset, h_data_classes = load_dataset(args, args.h_dataset, mode='h')
    h_num_classes = args.h_num_classes
    model = load_model(args.model, p_num_classes)
    # if args.p_dataset == 'tinyimagenet100':
    #     h_dataset = 'gtsrb'
    # elif args.p_dataset == 'gtsrb':
    #     h_dataset = 'cifar10'
    # elif args.p_dataset == 'cifar10':
    #     h_dataset = 'mnist'
    # elif args.p_dataset == 'svhn':
    #     h_dataset = 'cifar10'

    # model_path = f'results/global_model_client/{args.model}/{args.p_dataset}'
    # save_path = f'results/{args.date}/{args.train_mode}/{args.model}-{args.p_dataset}-{args.h_dataset}-{args.eta}-{args.normal_local_epochs}-{args.attacker_local_epochs}-{args.number_clients_total}-{args.number_clients_of_round}-{args.number_attackers_of_round}-{args.AR_attack_rounds}-{args.alpha}'
    # save_path = f'results/{args.date}/{args.train_mode}/{args.model}-{args.p_dataset}-{h_dataset}-{args.eta}-{args.normal_local_epochs}-{args.attacker_local_epochs}-{args.number_clients_total}-{args.number_clients_of_round}-{args.number_attackers_of_round}'
    # model_path += f'/{args.AR_attack_rounds - 1}.pth'
    model_path = f'results/global_model_clients/normal/{args.model}-{args.p_dataset}-{args.number_clients_of_round}/{args.AR_attack_rounds - 1}.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.to(args.device)
    model.eval()

    #这个模型只提取feature
    f_model = load_model(args.model, p_num_classes)
    f_model_path = f'results/global_model_clients/normal/{args.model}-{args.p_dataset}-{args.number_clients_of_round}/{args.test_round - 1}.pth'
    f_model.load_state_dict(torch.load(f_model_path, map_location=torch.device('cpu')))
    f_model = f_model.to(args.device)
    f_model.eval()

    # Training sort images and labels by class 
    train_sorted_images = [[] for _ in range(h_data_classes)]
    train_sorted_labels = [[] for _ in range(h_data_classes)]

    for image, label in h_trainset:
        if label < h_data_classes:
            train_sorted_images[label].append(image)
            train_sorted_labels[label].append(label)

    # 输出最大的k个样本，gstrb这种类别分布不均匀的用得到
    train_sorted_images_len = [len(train_sorted_images[i]) for i in range(len(train_sorted_images))]
    train_sorted_topk_index = sorted(range(len(train_sorted_images)), key=lambda i: train_sorted_images_len[i],
                                     reverse=True)[:h_num_classes]
    train_sorted_images = [train_sorted_images[i] for i in train_sorted_topk_index]
    train_sorted_labels = [train_sorted_labels[i] for i in train_sorted_topk_index]
    # relabel 为了简化后面求acc的操作
    train_sorted_labels = [[i] * train_sorted_images_len[train_sorted_topk_index[i]] for i in
                           range(len(train_sorted_labels))]

    if args.num_each_class > 0:
        each_class = int(args.num_each_class)
        train_sorted_images = [row[:each_class] for row in train_sorted_images]
        train_sorted_labels = [row[:each_class] for row in train_sorted_labels]

    # Testing sort images and labels by class
    test_sorted_images = [[] for _ in range(h_data_classes)]  # 10 classes in MNIST
    test_sorted_labels = [[] for _ in range(h_data_classes)]

    for image, label in h_testset:
        if label < h_data_classes:
            test_sorted_images[label].append(image)
            test_sorted_labels[label].append(label)

    test_sorted_images = [test_sorted_images[i] for i in train_sorted_topk_index]
    test_sorted_labels = [test_sorted_labels[i] for i in train_sorted_topk_index]
    # relabel
    if h_num_classes == p_num_classes:
        h_num_classes -= 1
    test_sorted_labels = [[i] * len(test_sorted_labels[i]) for i in range(len(train_sorted_labels))]
    time_logger = TimeLogger(args.logdir+ "/hijacking_time.csv")
    cor_map, neg_label = search_best_correspondence(args, model, train_sorted_images, p_num_classes, h_num_classes)
    # 先排序
    cor_map = dict(sorted(cor_map.items()))
    cor_map_v = list(cor_map.values())
    time_logger.log_time('category mapping')


    Features = extract_feature_maxtop1(model, p_num_classes, h_num_classes, cor_map, neg_label)
    time_logger.log_time('feature calcuation')
    unmasked_features = []
    masked_features = []
    MASK = []
    if args.train_mask==True:
        for class_index in range(h_num_classes):
            logx.msg(f'============hijack class {class_index} search=================')
            train_sorted_dataset = []
            # train_sorted_dataset.extend(list(zip(train_sorted_images[class_index], train_sorted_labels[class_index])))
            # train_sorted_loader = torch.utils.data.DataLoader(train_sorted_dataset, batch_size=args.batchsize, shuffle=True)

            train_sorted_dataset = Selected_DataSet(train_sorted_images, class_index)
            train_sorted_loader = torch.utils.data.DataLoader(train_sorted_dataset, batch_size=args.batchsize, shuffle=True)

            W = torch.randn((3, args.img_size, args.img_size)).to(args.device)
            W = W.detach()
            W.requires_grad = True

            # BCE = torch.nn.BCELoss()
            optimizer = torch.optim.Adam([W], lr=0.005, betas=(0.5, 0.999))
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
            # Example usage of sorted_loader:
            for i in range(args.train_round):
                for j, (images, images_neg, labels) in enumerate(train_sorted_loader):
                    batch_size = images.shape[0]

                    if images.shape[1] == 1:
                        images = np.tile(images, (1, 3, 1, 1))
                        images_neg = np.tile(images_neg, (1, 3, 1, 1))
                        X = torch.from_numpy(images).to(args.device)
                        X_neg = torch.from_numpy(images_neg).to(args.device)
                    else:
                        X = images.to(args.device)
                        X_neg = images_neg.to(args.device)

                    labels = torch.zeros(batch_size, h_num_classes).scatter_(1, labels.view(-1, 1), 1)
                    labels = labels.to(args.device)

                    P = torch.sigmoid(W)

                    X_adv = args.alpha * X + (1 - args.alpha) * P.expand_as(X)  # range [0, 1]
                    X_adv_neg = args.alpha * X_neg + (1 - args.alpha) * P.expand_as(X_neg)  # range [0, 1]

                    Y_adv = model.get_features(X_adv)
                    Y_adv_neg = model.get_features(X_adv_neg)

                    feature = Features[class_index].expand_as(Y_adv)
                    feature_neg = Features[-1].expand_as(Y_adv)

                    loss_pos = torch.norm(Y_adv - feature, p=2)
                    loss_neg = torch.norm(Y_adv_neg - feature_neg, p=2)

                    loss = loss_pos + args.beta * loss_neg

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    lr_scheduler.step()
                    if i == args.train_round-1 and args.save_feature==True:
                        x_feature = model.get_features(X)
                        cpu_feature = x_feature.detach().cpu().numpy()
                        unmasked_features.append(cpu_feature)

                        Y_adv_feature = Y_adv.detach().cpu().numpy()
                        masked_features.append(Y_adv_feature)

                    if (j + 1) % 10 == 0:
                        logx.msg('epoch %03d/%03d, batch %06d, loss %.6f' % (i + 1, 100, j + 1, loss.data.cpu().numpy()))
                avg_acc = 0
                logx.msg(f'============ hijack class {class_index} mask test=================')

                for _class_index in range(h_num_classes):
                    test_sorted_dataset = []
                    test_sorted_dataset.extend(
                        list(zip(test_sorted_images[_class_index], test_sorted_labels[_class_index])))
                    test_sorted_loader = torch.utils.data.DataLoader(test_sorted_dataset, batch_size=args.batchsize,
                                                                    shuffle=False)
                    acc = 0
                    for images, labels in test_sorted_loader:
                        if images.shape[1] == 1:
                            images = np.tile(images, (1, 3, 1, 1))
                            X = torch.from_numpy(images).to(args.device)
                        else:
                            X = images.to(args.device)
                        # cor_index = cor_map[class_index]
                        # labels = labels * 0 + cor_index
                        labels = labels * 0 + class_index

                        P = torch.sigmoid(W)
                        X_adv = args.alpha * X + (1 - args.alpha) * P.expand_as(X)  # range [0, 1]
                        Y_adv = f_model(X_adv)
                        Y_adv = F.softmax(Y_adv, 1)
                        out = Y_adv[:, cor_map_v]
                        pred = out.data.cpu().numpy().argmax(1)
                        acc += sum(labels.numpy() == pred) / float(len(labels) * len(test_sorted_loader))
                    logx.msg('class %02d, epoch %03d/%03d, test accuracy %.6f' % (_class_index, i, args.train_round, acc))
                    avg_acc += acc
                avg_acc /= h_num_classes
                # args.beta = avg_acc
            MASK.append(W)
        # save mask
        time_logger.log_time('mask computation')
        if args.save_feature == True:
            logx.msg('============Save Feature=================')
            with open(args.logdir+f'/{args.h_dataset}-{args.num_each_class}_features.pkl', 'wb') as file:
                save_to_file = {'unmaskd_features': unmasked_features, 'labels': train_sorted_labels,
                                'maskd_features': masked_features}
                pickle.dump(save_to_file, file)
            time_logger.log_time('feature save')
        
        if args.save_mask == True:
            logx.msg('============Save Mask=================')
            with open(args.logdir+f'/mask.pkl', 'wb') as file:
                save_to_file = {'mask':MASK}
                pickle.dump(save_to_file, file)
            time_logger.log_time('mask save')

    logx.msg('=============Normal Defence Test================')
    correct=0
    test_loader = torch.utils.data.DataLoader(p_testset, batch_size=1, shuffle=False)
    # acc = 0
    num_test = len(p_testset)
    defence_2 = defence_feature_squeezing(args,model)
    for images, labels in test_loader:
        if images.shape[1] == 1:
            images = np.tile(images, (1, 3, 1, 1))
            X = torch.from_numpy(images).to(args.device)
        else:
            X = images.to(args.device)
        if args.defence == 2:
            Y_adv = defence_2.defence_one_img(f_model,X)
        else:
            Y_adv = f_model(X)
        pred_label=Y_adv.argmax(dim=1)
        pred_label = pred_label.to('cpu')
        correct += (pred_label == labels).sum().item()
    pred_accuracy = 100 * correct / num_test
    logx.msg('original data accuracy: %.6f' % (pred_accuracy))
    logx.msg('============Final Test=================')
    prob_pred_count = 0
    max_diff_count = 0
    num_test = 0
    if args.train_mask ==False:
        with open(args.logdir+f'/mask.pkl' ,'rb') as file:
            load_data=pickle.load(file=file)
            MASK=load_data['mask']

    time_logger.log_time('hijacking data query start')
    for _class_index in range(h_num_classes):

        test_sorted_dataset = []
        test_sorted_dataset.extend(list(zip(test_sorted_images[_class_index], test_sorted_labels[_class_index])))
        test_sorted_loader = torch.utils.data.DataLoader(test_sorted_dataset, batch_size=1, shuffle=False)
        # acc = 0
        num_test += len(test_sorted_loader.dataset)
        for images, labels in test_sorted_loader:
            if images.shape[1] == 1:
                images = np.tile(images, (1, 3, 1, 1))
                X = torch.from_numpy(images).to(args.device)
            else:
                X = images.to(args.device)

            for mask_id in range(len(MASK)):
                W = MASK[mask_id]
                P = torch.sigmoid(W)
                X_adv = args.alpha * X + (1 - args.alpha) * P.expand_as(X)  # range [0, 1]

                X_adv_all_mask = X_adv if mask_id == 0 else torch.cat((X_adv_all_mask, X_adv), dim=0)

            Y_adv = f_model(X_adv_all_mask)
            Y_adv = F.softmax(Y_adv, 1)
            prob = Y_adv[:, cor_map_v]
            

            prob_max = prob.max(1, keepdim=True)[0].view(-1)
            prob_pred = prob_max.max(0, keepdim=True)[1].item()
            diff_values = torch.zeros(len(MASK), dtype=Y_adv.dtype).to(args.device)
            for mask_id in range(len(MASK)):
                diff_values[mask_id] += Y_adv[mask_id, mask_id] - Y_adv[mask_id, -1]
            max_diff_index = torch.argmax(diff_values).item()

            if prob_pred == labels.item():
                prob_pred_count += 1
            if max_diff_index == labels.item():
                max_diff_count += 1
    time_logger.log_time('hijacking data query end')
    pred_accuracy = 100 * prob_pred_count / num_test
    diff_accuracy = 100 * max_diff_count / num_test
    logx.msg('top1 prob accuracy %.6f, diff accuracy %.6f' % (pred_accuracy, diff_accuracy))

    logx.msg('============without mask Test=================')
    correct_count = 0
    num_test = 0
    for _class_index in range(h_num_classes):
        test_sorted_dataset = []
        test_sorted_dataset.extend(list(zip(test_sorted_images[_class_index], test_sorted_labels[_class_index])))
        test_sorted_loader = torch.utils.data.DataLoader(test_sorted_dataset, batch_size=1, shuffle=False)
        num_test += len(test_sorted_loader.dataset)

        for images, labels in test_sorted_loader:
            if images.shape[1] == 1:
                images = np.tile(images, (1, 3, 1, 1))
                X = torch.from_numpy(images).to(args.device)
            else:
                X = images.to(args.device)
            labels = labels.to(args.device)
            with torch.no_grad():
                outputs = f_model(X)[:,:10]
            _, pred = torch.max(outputs, 1)
            if pred.item() == labels.item():
                correct_count += 1
    final_accuracy = 100 * correct_count / num_test
    logx.msg('Test Accuracy: %.6f%%' % (final_accuracy))



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Decision-based Membership Inference Attack Toy Example')
    parser.add_argument('--date', type=str, default='1.30_defence3') # defence_2_demage
    parser.add_argument('--batchsize', nargs='+', default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.001 for adam; 0.1 for SGD)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=123213, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--logdir', type=str, default='results', help='target log directory')
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--img_size', type=int, default=64)
    ###############
    parser.add_argument('--train_mode', type=str, default='multi_mask', help='normal or ahmed or UAP or hidden or one_mask or multi_mask')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='resnet18, resent34, mobilenetv2, vgg16, preactresnet18, shufflenetv2') # preactresnet18用不了，loss一直是nan
    parser.add_argument('--p_dataset', type=str, default='cifar10', help='primary dataset: cifar10, tinyimagenet100, logos10, logos6')
    parser.add_argument('--h_dataset', type=str, default='gtsrb', help='hijacking dataset: mnist, gtsrb')
    parser.add_argument('--p_num_classes', type=int,default=10)
    parser.add_argument('--h_num_classes', type=int,default=9)  # 2,4,6,8,10
    parser.add_argument('--eta', type=float, default=1, help='fixed') # 影响客户端模型聚合权重的参数
    parser.add_argument('--normal_local_epochs', type=int, default=2, help='fixed')
    parser.add_argument('--attacker_local_epochs', type=int, default=2, help='fixed')
    parser.add_argument('--number_clients_total', type=int, default=50, help='fixed')
    parser.add_argument('--number_clients_of_round', type=int, default=5, help='fixed')
    parser.add_argument('--number_attackers_of_round', type=int, default=1,
                        help='must 0 if train_mode == normal; must >0 if train_mode==ahmed or Fusion')
    parser.add_argument('--attack_rounds', nargs='+', default=[150], help='default 150')
    parser.add_argument('--attacker_lr', type=float, default=0.05, help='fixed')
    parser.add_argument('--eugene_scaling', type=int, default=10, help='fixed') #影响的是baseline
    parser.add_argument('--flhj_scaling', type=float, default=0.0, help='fixed')
    parser.add_argument('--UAP_delta', type=float, default=0.1)
    parser.add_argument('--fgsm_eps', type=float, default=0.01)
    parser.add_argument('--uap_max_iter', type=int, default=10)
    parser.add_argument('--uap_eps', type=float, default=5.5)
    parser.add_argument('--ar_lmd', type=float, default=5e-5)
    parser.add_argument('--test_round', type=int, default=200, help='final global model') # fixed
    parser.add_argument('--AR_attack_rounds', type=int, default=150, help='default 150') # 20 60 80 100 150
    parser.add_argument('--alpha', type=float, default=0.5) # 控制mask和原始样本的混合比例 默认是0.5
    parser.add_argument('--beta', type=float, default=1) # 控制负例权重，默认是1.2
    parser.add_argument('--train_round', type=int, default=100)
    parser.add_argument('--num_each_class', type=int, default=-1) # 小于0无效,实验100 300 500 700 full
    parser.add_argument('--save_feature', default=True, help='if save feature as pkl')
    parser.add_argument('--save_mask',default=True,help='true:save mask as pkl')
    parser.add_argument('--train_mask',default=False,help='if train mask')

    parser.add_argument('--color_deep',default=8)
    parser.add_argument('--smooth_type',default='local',help='local or non_local')
    parser.add_argument('--defence2_thr',default=50,type=int,help='60 在imagenet100上降低1%')

    parser.add_argument('--gpu',default=1)
    args = parser.parse_args()
    fixed_seed(args.seed)
    check_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.train_mode=='normal':
        args.logdir = f'{args.logdir}/{args.date}/{args.train_mode}/{args.model}-{args.p_dataset}-{args.number_clients_of_round}'
    else:
        args.logdir = f'{args.logdir}/{args.date}/{args.train_mode}/{args.model}-{args.p_dataset}-{args.h_dataset}-es-{args.eugene_scaling}-{args.eta}-{args.normal_local_epochs}-{args.attacker_local_epochs}-{args.number_clients_total}-{args.number_clients_of_round}-{args.number_attackers_of_round}-a{args.AR_attack_rounds}-t{args.test_round}-{args.alpha}-{args.num_each_class}-class-{args.p_num_classes}-{args.h_num_classes}'
    print(args.logdir)
    if args.train_mode in ['normal']:
        fl_run(args)
    if args.train_mode == 'multi_mask':
        AR_search_interplation_feature(args)
    else:
        print('run nothing')