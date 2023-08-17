from ast import Global
import imp
import logging
import pstats
from readline import append_history_file
from turtle import st
from typing import Dict
import sys, math
import torch
from copy import deepcopy
import numpy as np
from models.model import Model
# from models.nc_model import NCModel
# from synthesizers.synthesizer import Synthesizer
from losses.loss_functions import compute_all_losses_and_grads
from utils.min_norm_solvers import MGDASolver
from utils.parameters import Params
import copy
from torch.nn import functional as F
from models.converter import train_converter
import torch.nn as nn
import random
# from test import test
logger = logging.getLogger('logger')
# from time import *
from torchvision.utils import save_image
class Attack:
    params: Params
    # synthesizer: Synthesizer
    nc_model: Model
    nc_optim: torch.optim.Optimizer
    loss_hist = list()
    # fixed_model: Model

    def __init__(self, params):
        self.params = params
        # self.synthesizer = synthesizer
    
    def compute_hijack_loss(self, model, criterion, clean_batch, hidden_batch, attack):
        """
        :param model:
        :param criterion:
        :param batch:
        :param attack: Do not attack at all. Ignore all the parameters
        :return:
        """
        clean_batch = clean_batch.clip(self.params.clip_batch)
        hidden_batch = hidden_batch.clip(self.params.clip_batch)
        loss_tasks = self.params.loss_tasks.copy() if attack else ['normal']
        # batch_hijack = self.synthesizer.make_hijack_batch(batch, attack=attack)
        batch_hijack = None
        scale = dict()

        # if 'neural_cleanse' in loss_tasks:
        #     self.neural_cleanse_part1(model, batch, batch_back)

        # if self.params.loss_threshold and (np.mean(self.loss_hist) >= self.params.loss_threshold
        #                                    or len(self.loss_hist) < 1000):
        #     loss_tasks = ['normal']

        if len(loss_tasks) == 1:
            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, clean_batch, hidden_batch, compute_grad=False
            )

        elif self.params.loss_balance == 'MGDA':

            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, clean_batch, hidden_batch, compute_grad=True)
            if len(loss_tasks) > 1:
                scale = MGDASolver.get_scales(grads, loss_values,
                                              self.params.mgda_normalize,
                                              loss_tasks)

        elif self.params.loss_balance == 'fixed':
            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, clean_batch, hidden_batch, compute_grad=False)

            for t in loss_tasks:
                scale[t] = self.params.fixed_scales[t]
        else:
            raise ValueError(f'Please choose between `MGDA` and `fixed`.')

        if len(loss_tasks) == 1:
            scale = {loss_tasks[0]: 1.0}
        self.loss_hist.append(loss_values['normal'].item())
        self.loss_hist = self.loss_hist[-1000:]
        hijack_loss = self.scale_losses(loss_tasks, loss_values, scale)
        return hijack_loss

    def scale_losses(self, loss_tasks, loss_values, scale):
        hijack_loss = 0
        for it, t in enumerate(loss_tasks):
            self.params.running_losses[t].append(loss_values[t].item())
            self.params.running_scales[t].append(scale[t])
            if it == 0:
                hijack_loss = scale[t] * loss_values[t]
            else:
                hijack_loss += scale[t] * loss_values[t]
        self.params.running_losses['total'].append(hijack_loss.item())
        return hijack_loss

    def neural_cleanse_part1(self, model, batch, batch_back):
        self.nc_model.zero_grad()
        model.zero_grad()

        self.nc_model.switch_grads(True)
        model.switch_grads(False)
        output = model(self.nc_model(batch.inputs))
        nc_tasks = ['neural_cleanse_part1', 'mask_norm']

        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        loss_values, grads = compute_all_losses_and_grads(nc_tasks,
                                                          self, model,
                                                          criterion, batch,
                                                          batch_back,
                                                          compute_grad=False)
        # Using NC paper params
        logger.info(loss_values)
        loss = 0.999 * loss_values['neural_cleanse_part1'] + 0.001 * loss_values['mask_norm']
        loss.backward()
        self.nc_optim.step()

        self.nc_model.switch_grads(False)
        model.switch_grads(True)

    def fl_scale_update(self, weight_scale, local_update: Dict[str, torch.Tensor]):
        for name, value in local_update.items():
            value.mul_(weight_scale)

######### Oakland #########
######### Oakland #########

def itr_merge(itrs):
    for itr in itrs:
        for v in itr:
            yield v

def Select_Greedy_Loss(ENTROPY):
    min_entropy_class_summary = []
    while len(ENTROPY.keys()) != 0:
        min_entropy_class = [10000, -1, -1] # entropy, relabel class, orignal class

        for k in ENTROPY.keys():
            # print(k, ENTROPY[k])
            cur_min_entropy_class = min(zip(ENTROPY[k].values(), ENTROPY[k].keys()))

            if cur_min_entropy_class[0] < min_entropy_class[0]:
                min_entropy_class[0] = cur_min_entropy_class[0]
                min_entropy_class[1] = cur_min_entropy_class[1]
                min_entropy_class[2] = k

        min_entropy_class_summary.append(min_entropy_class)
        ENTROPY.pop(min_entropy_class[2])
        for k in ENTROPY.keys():
            ENTROPY[k].pop(min_entropy_class[1])
    return min_entropy_class_summary


def Select_Brute_Loss(ENTROPY):
    min_loss = 10000    
    min_entropy_class_summary = []

    def compute_loss(record):
        loss = 0
        for original_label, target_label in enumerate(record):
            loss += ENTROPY[original_label][target_label]
        return loss
    
    def mapping(record):
        min_entropy_class_summary = []
        for original_label, target_label in enumerate(record):
            min_entropy_class = [0, target_label, original_label]
            min_entropy_class_summary.append(min_entropy_class)
        return min_entropy_class_summary

    record = []
    reference = list(range(10))
    sys.setrecursionlimit(200000)
    def recursion(record, min_entropy_class_summary, min_loss):
        for i in reference:
            if i in record:
                continue
            else:
                record.append(i)
                if len(record) == 10:
                    loss = compute_loss(record)
                    if loss < min_loss:
                        min_loss = loss
                        min_entropy_class_summary = mapping(record)
                    record.append(loss)
                    record = record[0:len(record)-2]
                    return record, min_entropy_class_summary, min_loss
            record, min_entropy_class_summary, min_loss = recursion(record, min_entropy_class_summary, min_loss)
            record = record[0:len(record)-1]
        return record, min_entropy_class_summary, min_loss
    _, min_entropy_class_summary, _ = recursion(record, min_entropy_class_summary, min_loss)
    return min_entropy_class_summary

def assign_lable(extra_info, H_idx, y):
    # H_idx = 0
    for idx in range(y.shape[0]):
        if idx == 0:
            label = y[idx:idx+1] * 0 + extra_info[H_idx][y[idx:idx+1].item()]
        else:
            label = torch.cat((label, y[idx:idx+1] * 0 + extra_info[H_idx][y[idx:idx+1].item()]), 0)
    return label

def search_label_mapping(model, H_train_loader, hlpr):
    batch_id = 0
    for x, y in itr_merge([H_train_loader]):
        input = x.to('cpu') if batch_id == 0 else torch.cat((input, x.to('cpu')), 0)
        label = y.to('cpu') if batch_id == 0 else torch.cat((label, y.to('cpu')), 0)
        batch_id = batch_id + 1
    concat_dataset = torch.utils.data.TensorDataset(input, label)
    H_train_loader = torch.utils.data.DataLoader(concat_dataset, batch_size=1, shuffle=False, num_workers=0)
    if hlpr.params.H_dataset[0] in ['mnist', 'emnist', 'kmnist', 'fashion_mnist', 'cifar10', 'stl10']:
        num_classes = 10
    elif hlpr.params.H_dataset[0] in ['place20']:
        num_classes = 20
    elif hlpr.params.H_dataset[0] in ['place40']:
        num_classes = 40
    elif hlpr.params.H_dataset[0] in ['cifar100']:
        num_classes = 100
    elif hlpr.params.H_dataset[0] in ['tinyimagenet']:
        num_classes = 200
    entropy_base = dict(zip([x for x in range(num_classes)], [x for x in range(num_classes)]))
    num_base = dict(zip([0], [0]))
    ENTROPY_NUM = copy.deepcopy(entropy_base)
    ENTROPY = copy.deepcopy(entropy_base)
    for i in range(num_classes):
        ENTROPY_NUM[i] = copy.deepcopy(num_base)
    for i in range(num_classes):
        ENTROPY[i] = copy.deepcopy(ENTROPY_NUM)
    for input, label in itr_merge([H_train_loader]):
        input, label = input.to(hlpr.params.device), label.to(hlpr.params.device)
        logits = model(input)
        logits = logits[:, :10]
        for i in range(num_classes):
            entropy_target = label * 0 + i                
            for k in ENTROPY[label.item()][i].keys():
                num = k + 1
                ENTROPY[label.item()][i].update({num:ENTROPY[label.item()][i].pop(k)+F.cross_entropy(logits, entropy_target).item()})
                break
    for K in ENTROPY.keys():
        for k in ENTROPY[K].keys():
            for num in ENTROPY[K][k].keys():
                ENTROPY[K][k] = ENTROPY[K][k][num] / num
    if hlpr.params.greedy_brute == 'greedy':
        logger.info('=========== Greedy_Search Start ===========')
        min_entropy_class_summary = Select_Greedy_Loss(ENTROPY)
        logger.info('=========== Greedy_Search End ===========')
    elif hlpr.params.greedy_brute == 'brute':
        logger.info('=========== Brute_Search Start ===========')
        min_entropy_class_summary = Select_Brute_Loss(ENTROPY)
        logger.info('=========== Brute_Search End ===========')
    origin_labels = [entropy_class_summary[2] for entropy_class_summary in min_entropy_class_summary]
    target_labels = [entropy_class_summary[1] for entropy_class_summary in min_entropy_class_summary]
    return origin_labels, target_labels
def apply_converter(hlpr, Converter, model, relabeled_loader):
    if hlpr.params.converter_multi_updating == 0:
        train_converter(Converter, model, relabeled_loader, hlpr) 
        hlpr.params.converter_multi_updating = 2
    elif hlpr.params.converter_multi_updating == 1:
        train_converter(Converter, model, relabeled_loader, hlpr) 
    Converter.eval()
    batch_id = 0
    for x, y in itr_merge([relabeled_loader]):
        x = x.to(hlpr.params.device)
        x_T = Converter(x)
        x_T = x_T.to(torch.device('cpu'))
        if batch_id == 0:
            input = x_T
            label = y
        else:
            input = torch.cat((input, x_T), 0)
            label = torch.cat((label, y), 0)
        batch_id = batch_id + 1
    converted_dataset = torch.utils.data.TensorDataset(input.detach(), label.type(torch.long).detach())
    converted_loader = torch.utils.data.DataLoader(converted_dataset, batch_size= hlpr.params.batch_size, shuffle=False, num_workers=0)
    return converted_loader
def trainloader_in_FL(user, hlpr, local_epoch, model, Converter):
    if user.compromised:
        model.eval()
        H_train_loaders = []
        if hlpr.params.hijack_train_setting in ['naive']:
            for H_idx, (H_train_loader, H_dataset_size) in enumerate(zip(user.H_train_loaders, hlpr.params.H_dataset_size)):
                # if hlpr.params.converted_data == 'no_converter':
                H_train_loaders.append(H_train_loader)
                user.H_train_loaders = H_train_loaders
                # elif hlpr.params.converted_data in ['not_joining', 'joining']:
                #     converted_loader = apply_converter(hlpr, Converter, model, H_train_loader)
                #     H_train_loaders.append(converted_loader) 
                # hlpr.params.h_test_data_relabeled_or_converted = False                   
        elif hlpr.params.hijack_train_setting == 'relabling':
            for H_idx, (H_train_loader, H_dataset_size) in enumerate(zip(user.H_train_loaders, hlpr.params.H_dataset_size)):
                if len(user.mapping_func) == H_idx:
                    origin_labels, target_labels = search_label_mapping(model, H_train_loader, hlpr)
                    user.mapping_func.append(dict(zip(origin_labels, target_labels)))
                if not user.relabeled:
                    batch_id = 0
                    for x, y in itr_merge([H_train_loader]):
                        if batch_id == 0:
                            input = x
                            label = assign_lable(user.mapping_func, H_idx, y)
                        else:
                            input = torch.cat((input, x), 0)
                            label = torch.cat((label, assign_lable(user.mapping_func, H_idx, y)), 0)
                        batch_id = batch_id + 1
                    relabeled_dataset = torch.utils.data.TensorDataset(input, label.type(torch.long))
                    relabeled_loader = torch.utils.data.DataLoader(relabeled_dataset, batch_size=hlpr.params.batch_size, shuffle=False, num_workers=0)
                    if hlpr.params.converted_data == 'no_converter':
                        H_train_loaders.append(relabeled_loader)
                    elif hlpr.params.converted_data in ['not_joining', 'joining']:
                        converted_loader = apply_converter(hlpr, Converter, model, relabeled_loader)
                        # if hlpr.params.delay_attack:
                        #     if hlpr.params.converted_data == 'joining':
                        #         hlpr.params.converted_data = 'joining_delay'
                        #     elif hlpr.params.converted_data == 'joining_delay':
                        #         hlpr.params.converted_data = 'joining'
                        #         hlpr.params.hijack_local_epochs = 5
                        # if hlpr.params.converted_data == 'not_joining':
                        #     pass
                        # elif hlpr.params.converted_data == 'joining' and hlpr.params.converter_epochs == 100:
                        #     hlpr.params.converted_data = 'joining_10'
                        #     hlpr.params.converter_epochs = 5
                        # elif hlpr.params.converted_data == 'joining_10' and hlpr.params.converter_epochs == 5:
                        #     hlpr.params.converted_data = 'joining'
                        #     hlpr.params.hijack_local_epochs = 1

                        H_train_loaders.append(converted_loader)
                else:
                    H_train_loaders.append(H_train_loader)
                user.H_train_loaders = H_train_loaders
                user.relabeled = True
                hlpr.params.h_test_data_relabeled_or_converted = False
        elif hlpr.params.hijack_train_setting == 'no_relabling' and hlpr.params.h_test_data_relabeled_or_converted:
            for H_idx, (H_train_loader, H_dataset_size) in enumerate(zip(user.H_train_loaders, hlpr.params.H_dataset_size)):
                # if len(user.mapping_func) == H_idx:
                #     origin_labels, target_labels = search_label_mapping(model, H_train_loader, hlpr)
                #     user.mapping_func.append(dict(zip(origin_labels, target_labels)))
                # if not user.relabeled:
                batch_id = 0
                for x, y in itr_merge([H_train_loader]):
                    if batch_id == 0:
                        input = x
                        label = y
                    else:
                        input = torch.cat((input, x), 0)
                        label = torch.cat((label, y), 0)
                    batch_id = batch_id + 1
                relabeled_dataset = torch.utils.data.TensorDataset(input, label.type(torch.long))
                relabeled_loader = torch.utils.data.DataLoader(relabeled_dataset, batch_size=hlpr.params.batch_size, shuffle=False, num_workers=0)
                if hlpr.params.converted_data == 'no_converter':
                    H_train_loaders.append(relabeled_loader)
                elif hlpr.params.converted_data in ['not_joining', 'joining']:
                    converted_loader = apply_converter(hlpr, Converter, model, relabeled_loader)
                    # if hlpr.params.converted_data == 'not_joining':
                    #     pass
                    # elif hlpr.params.converted_data == 'joining' and hlpr.params.converter_epochs == 100:
                    #     hlpr.params.converted_data = 'joining_10'
                    #     hlpr.params.converter_epochs = 5
                    # elif hlpr.params.converted_data == 'joining_10' and hlpr.params.converter_epochs == 5:
                    #     hlpr.params.converted_data = 'joining'
                    #     hlpr.params.hijack_local_epochs = 1

                    H_train_loaders.append(converted_loader)
                # else:
                #     H_train_loaders.append(H_train_loader)
                user.H_train_loaders = H_train_loaders
                # user.relabeled = True
                hlpr.params.h_test_data_relabeled_or_converted = False
        # if hlpr.params.hijack_train_setting in ['naive_old']:
        #     # if hlpr.params.converted_data in ['no_converter', 'not_joining']:
        #     #     combine_data_loader = user.A_train_loader
        #     #     return train_loader, None
        #     # elif hlpr.params.converted_data in ['joining']:
        #     combine_data_loader = user.A_train_loader + H_train_loaders
        #     batch_id = 0
        #     for x, y in itr_merge(combine_data_loader):
        #         input = x.to('cpu') if batch_id == 0 else torch.cat((input, x.to('cpu')), 0)
        #         label = y.to('cpu') if batch_id == 0 else torch.cat((label, y.to('cpu')), 0)
        #         batch_id = batch_id + 1
        #     concat_dataset = torch.utils.data.TensorDataset(input, label)
        #     train_loader = torch.utils.data.DataLoader(concat_dataset, batch_size= hlpr.params.batch_size, shuffle=False, num_workers=0)
        #     # train_loader = user.A_train_loader[0]
        #     return train_loader, None
        if hlpr.params.hijack_train_setting in ['naive'] : # and hlpr.params.A_dataset != 'celeba'
            batch_id = 0
            for x, y in itr_merge(user.A_train_loader+user.H_train_loaders):
                input = x.to('cpu') if batch_id == 0 else torch.cat((input, x.to('cpu')), 0)
                label = y.to('cpu') if batch_id == 0 else torch.cat((label, y.to('cpu')), 0)
                batch_id = batch_id + 1
            train_dataset = torch.utils.data.TensorDataset(input, label)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= hlpr.params.batch_size, shuffle=False, num_workers=0)
            return train_loader, None
        
        elif hlpr.params.hijack_train_setting in ['relabling', 'no_relabling'] : # or (hlpr.params.hijack_train_setting in ['naive'] and hlpr.params.A_dataset == 'celeba')
            batch_id = 0
            for x, y in itr_merge(user.A_train_loader):
                input = x.to('cpu') if batch_id == 0 else torch.cat((input, x.to('cpu')), 0)
                label = y.to('cpu') if batch_id == 0 else torch.cat((label, y.to('cpu')), 0)
                batch_id = batch_id + 1
            train_dataset = torch.utils.data.TensorDataset(input, label)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= hlpr.params.batch_size, shuffle=False, num_workers=0)

            batch_id = 0
            for x, y in itr_merge(user.H_train_loaders):
                input = x.to('cpu') if batch_id == 0 else torch.cat((input, x.to('cpu')), 0)
                label = y.to('cpu') if batch_id == 0 else torch.cat((label, y.to('cpu')), 0)
                batch_id = batch_id + 1

            H_train_dataset = torch.utils.data.TensorDataset(input, label)
            # H_train_loader = torch.utils.data.DataLoader(H_train_dataset, batch_size=local_dataset_size, shuffle=False, num_workers=0)

            # # train_loader = user.A_train_loader[0]
            batch_num = len(train_dataset)/hlpr.params.batch_size
            batch_num = math.ceil(batch_num)
            H_batch_size = len(H_train_dataset)/batch_num
            H_batch_size = math.ceil(H_batch_size)
            if H_batch_size > 500:
                H_batch_size = 512
            H_train_loader = torch.utils.data.DataLoader(H_train_dataset, batch_size= H_batch_size, shuffle=False, num_workers=0)
            # print(batch_num, H_batch_size)
            # if H_batch_size == 512:
            #     batch_num = len(H_train_dataset)/H_batch_size
            #     batch_num = math.ceil(batch_num)
            #     A_batch_size = len(train_dataset)/batch_num
            #     A_batch_size = math.ceil(A_batch_size)
            #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=A_batch_size, shuffle=False, num_workers=0)

            # print(batch_num, A_batch_size)

            print(len(train_loader), len(H_train_loader))
            return train_loader, H_train_loader
    else:
        train_loader = user.A_train_loader[0]
        return train_loader, None
        
def testloader_in_FL(hlpr, hijack, extra_info, Converter, A_test_loaders=None, H_test_loaders=None):
    Converter.eval()
    if hijack:
        if hlpr.params.hijack_train_setting in ['normal', 'naive']:
            H_test_loaders = hlpr.task.fl_H_tasks_test_loaders
        elif hlpr.params.hijack_train_setting in ['relabling', 'no_relabling'] and (not hlpr.params.h_test_data_relabeled_or_converted):
            H_test_loaders = []
            for H_idx, H_tasks_test_loader in enumerate(hlpr.task.fl_H_tasks_test_loaders):
                if hlpr.params.converted_data == 'no_converter':
                    # H_test_loaders.append(H_tasks_test_loader)
                    batch_id = 0
                    for x, y in itr_merge([H_tasks_test_loader]):
                        # x = x.to(hlpr.params.device)
                        # x = Converter(x)
                        x = x.to(torch.device('cpu'))
                        y = y.to(torch.device('cpu'))
                        if batch_id == 0:
                            input = x
                            label = assign_lable(extra_info, H_idx, y) if hlpr.params.hijack_train_setting == 'relabling' else y
                        else:
                            input = torch.cat((input, x), 0)
                            label = torch.cat((label, assign_lable(extra_info, H_idx, y)), 0) if hlpr.params.hijack_train_setting == 'relabling' else torch.cat((label, y), 0)
                        batch_id = batch_id + 1
                    sub_dataset = torch.utils.data.TensorDataset(input, label)
                    H_test_loaders.append(torch.utils.data.DataLoader(sub_dataset, batch_size= hlpr.params.batch_size, shuffle=False, num_workers=0))
                elif hlpr.params.converted_data in ['not_joining', 'joining'] or 'joining_' in hlpr.params.converted_data:
                    batch_id = 0
                    for x, y in itr_merge([H_tasks_test_loader]):
                        x = x.to(hlpr.params.device)
                        
                        x = Converter(x)
                        x = x.to(torch.device('cpu'))
                        y = y.to(torch.device('cpu'))
                        if batch_id == 0:
                            input = x
                            label = assign_lable(extra_info, H_idx, y) if hlpr.params.hijack_train_setting == 'relabling' else y
                        else:
                            input = torch.cat((input, x), 0)
                            label = torch.cat((label, assign_lable(extra_info, H_idx, y)), 0) if hlpr.params.hijack_train_setting == 'relabling' else torch.cat((label, y), 0)
      
                        exit()
                        
                        batch_id = batch_id + 1
                    sub_dataset = torch.utils.data.TensorDataset(input, label)
                    H_test_loaders.append(torch.utils.data.DataLoader(sub_dataset, batch_size= hlpr.params.batch_size, shuffle=False, num_workers=0))
            hlpr.params.h_test_data_relabeled_or_converted = True
        else:
            H_test_loaders = hlpr.task.fl_H_tasks_test_loaders if len(H_test_loaders) == 0 else H_test_loaders # if 只在第一次被指定fl_H_tasks_test_loaders， 后面只有else, 要么是fl_H_tasks_test_loaders要么relabelling的
    else:
        A_test_loaders = [hlpr.task.A_test_loader]
    return A_test_loaders, H_test_loaders

