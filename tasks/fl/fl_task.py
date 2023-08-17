import enum
import imp
import math
import random
from copy import deepcopy
from turtle import ycor
from typing import List, Any, Dict

from metrics.accuracy_metric import AccuracyMetric
from metrics.test_loss_metric import TestLossMetric
from tasks.fl.fl_user import FLUser
import torch
import logging
from torch.nn import Module
# from helper import Helper
from tasks.task import Task
logger = logging.getLogger('logger')
from models.converter import UnetGenerator, converter_2, converter_4, converter_6, converter_8, converter_10


def build_converter(converter_layer): # 'converter_2, converter_4, converter_6, converter_8'
    if converter_layer == 'converter_2':
        Transformer = converter_2()
    elif converter_layer == 'converter_4':
        Transformer = converter_4()
    elif converter_layer == 'converter_6':
        Transformer = converter_6()
    elif converter_layer == 'converter_8':
        Transformer = converter_8()
    elif converter_layer == 'converter_10':
        Transformer = converter_10()
    return Transformer
class FederatedLearningTask(Task):
    fl_train_loaders: List[Any] = None
    ignored_weights = ['num_batches_tracked']#['tracked', 'running']
    adversaries: List[int] = None

    def init_task(self):
        
        self.load_data()
        self.model = self.build_model()
        self.resume_model()
        self.model = self.model.to(self.params.device)

        self.local_model = self.build_model().to(self.params.device)
        self.criterion = self.make_criterion()
        self.adversaries = self.sample_adversaries()

        self.metrics = [AccuracyMetric(), TestLossMetric(self.criterion)]
        self.set_input_shape()
        return

    def get_empty_accumulator(self):
        weight_accumulator = dict()
        for name, data in self.model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)
        return weight_accumulator

    def sample_users_for_round(self, epoch) -> List[FLUser]:
        single_epoch_attack = self.params.fl_single_epoch_attack
        multi_epoch_attack = self.params.fl_multi_epoch_attack
        multi_user_attack = self.params.fl_multi_user_attack

        # sampled_ids = random.sample(range(self.params.fl_total_participants), self.params.fl_no_models)
        # random.shuffle(all_range)
        epoch_id = int((epoch-1) % (self.params.fl_total_participants/self.params.fl_no_models))
        sampled_ids = list(range(self.params.fl_total_participants))[epoch_id*self.params.fl_no_models: (epoch_id+1)*self.params.fl_no_models]
        sampled_users = []
        for pos, user_id in enumerate(sampled_ids):
            # if self.params.hijack_train_setting == 'fake_benign' and user_id >= self.params.fl_number_of_adversaries and user_id < int(self.params.fl_number_of_adversaries + self.params.hijack_fake_benign):
            #     A_train_loader = [self.fl_A_train_loaders[(user_id-self.params.fl_number_of_adversaries)%self.params.fl_number_of_adversaries], self.fl_A_train_loaders[user_id]]
            #     compromised = False
            # else:
            A_train_loader = [self.fl_A_train_loaders[user_id]]
            compromised = self.check_user_compromised(epoch, pos, user_id)
            H_train_loaders = []
            Transformers = []

            if compromised:
                if (single_epoch_attack == 0) and (multi_epoch_attack == 0) and (multi_user_attack):
                    attacker_id = self.adversaries.index(user_id)
                else:
                    attacker_id = pos
      
                if (len(self.params.H_dataset)>1) and (self.params.fl_number_of_adversaries % len(self.fl_H_tasks_train_loaders) == 0):
                    adv_clusters = self.params.fl_number_of_adversaries / len(self.fl_H_tasks_train_loaders)  # 6/2 = 3
                    task_id = int(attacker_id/ adv_clusters)
                    subdata_id = int(attacker_id % adv_clusters)
                    H_train_loaders.append(self.fl_H_tasks_train_loaders[task_id][subdata_id])
                    Transformers.append(build_converter(self.params.converter_layer))
                elif (len(self.params.H_dataset)>1) and (len(self.fl_H_tasks_train_loaders) % self.params.fl_number_of_adversaries == 0):
                    task_clusters = len(self.fl_H_tasks_train_loaders) / self.params.fl_number_of_adversaries # 8/4 = 2
                    # print('task_clusters', task_clusters)
                    for H_task_id, H_task_loader in enumerate(self.fl_H_tasks_train_loaders):
                        if int(H_task_id / task_clusters) == attacker_id:
                            # subdata_id = H_task_id % task_clusters
                            H_train_loaders.append(H_task_loader[0])
                            Transformers.append(build_converter(self.params.converter_layer))
                else:
                    for H_loader in self.fl_H_tasks_train_loaders:
                        H_train_loaders.append(H_loader[attacker_id])
                        Transformers.append(build_converter(self.params.converter_layer))

                # print(A_train_loader)

                # batch_id = 0
                # for x, y in itr_merge(A_train_loader, H_train_loader):
                #     input = x if batch_id == 0 else torch.cat((input, x), 0)
                #     label = y if batch_id == 0 else torch.cat((label, y), 0)
                #     batch_id = batch_id + 1
                # concat_dataset = torch.utils.data.TensorDataset(input, label)
                # A_train_loader = torch.utils.data.DataLoader(concat_dataset, batch_size= self.params.batch_size, shuffle=True, num_workers=0)

            user = FLUser(user_id, compromised=compromised, A_train_loader=A_train_loader, mapping_func = [], H_train_loaders=H_train_loaders if compromised else None, transformer=Transformers if compromised else None)
            sampled_users.append(user)
            # print(user)
        return sampled_users
        
    def check_user_compromised(self, epoch, pos, user_id):
        """Check if the sampled user is compromised for the attack.
        If single_epoch_attack is defined (eg not None) then ignore
        :param epoch:
        :param pos:
        :param user_id:
        :return:
        """
        compromised = False
        number_of_adversaries = self.params.fl_number_of_adversaries
        single_epoch_attack = self.params.fl_single_epoch_attack
        multi_epoch_attack = self.params.fl_multi_epoch_attack
        multi_user_attack = self.params.fl_multi_user_attack
        if number_of_adversaries == 0:
            compromised = False
        elif (single_epoch_attack == 0) and (multi_epoch_attack == 0) and (multi_user_attack):
            compromised = user_id in self.adversaries
            if compromised:
                logger.warning(f'Attacking by multi_user_attack regardless of epochs. Compromised'
                            f' user: {user_id}.')
        elif (single_epoch_attack > 0) and (multi_epoch_attack == 0) and (not multi_user_attack):
            if epoch == single_epoch_attack:
                if pos < number_of_adversaries:
                    compromised = True
                    logger.warning(f'Attacking once at epoch {epoch}. Compromised'
                                    f' user: {user_id}.')
        elif (single_epoch_attack == 0) and (multi_epoch_attack > 0) and (not multi_user_attack):
            if epoch in list(range(1, self.params.epochs, multi_epoch_attack)):
                if pos < number_of_adversaries:
                        compromised = True
                        logger.warning(f'Attacking multi times at evary {multi_epoch_attack}. Compromised'
                                    f' user: {user_id}.')
        return compromised

    def sample_adversaries(self) -> List[int]:
        adversaries_ids = []
        number_of_adversaries = self.params.fl_number_of_adversaries
        single_epoch_attack = self.params.fl_single_epoch_attack
        multi_epoch_attack = self.params.fl_multi_epoch_attack
        multi_user_attack = self.params.fl_multi_user_attack

        if number_of_adversaries == 0:
            logger.warning(f'Running vanilla FL, no attack.')
        elif (single_epoch_attack == 0) and (multi_epoch_attack == 0) and (multi_user_attack):
            adversaries_ids = random.sample(range(self.params.fl_total_participants), number_of_adversaries)
            logger.warning(f'Attacking over multiple epochs with following '
                           f'users compromised: {adversaries_ids}.')
        elif (single_epoch_attack > 0) and (multi_epoch_attack == 0) and (not multi_user_attack):
            logger.warning(f'Attack only on epoch: '
                           f'{single_epoch_attack} with '
                           f'{number_of_adversaries} compromised'
                           f' users.')
        elif (single_epoch_attack == 0) and (multi_epoch_attack > 0) and (not multi_user_attack):
            logger.warning(f'Attack on every '
                           f'{multi_epoch_attack} epoch with '
                           f'{number_of_adversaries} compromised'
                           f' users.')

        return adversaries_ids

    def check_user_compromised_old(self, epoch, pos, user_id):
        """Check if the sampled user is compromised for the attack.

        If single_epoch_attack is defined (eg not None) then ignore
        :param epoch:
        :param pos:
        :param user_id:
        :return:
        """
        compromised = False
        if self.params.fl_single_epoch_attack > 0: #attack on single epoch, the pos_id is the attacker
            if epoch == self.params.fl_single_epoch_attack:
                if pos < self.params.fl_number_of_adversaries:
                    compromised = True
                    logger.warning(f'Attacking once at epoch {epoch}. Compromised'
                                   f' user: {user_id}.')
        elif self.params.fl_multi_epoch_attack > 0:
            if self.params.fl_multi_epoch_attack == 0:
                if pos < self.params.fl_number_of_adversaries:
                        compromised = True
                        logger.warning(f'Attacking at every epoch. Compromised'
                                    f' user: {user_id}.')
            elif epoch / self.params.fl_multi_epoch_attack == 0:
                if pos < self.params.fl_number_of_adversaries:
                        compromised = True
                        logger.warning(f'Attacking at every {epoch} epoch. Compromised'
                                    f' user: {user_id}.')
        elif self.params.fl_multi_user_attack:
            compromised = user_id in self.adversaries # attack on the selected attackers in self.adversaries, not gurrantee every epoch.

        return compromised

    def sample_adversaries_old(self) -> List[int]:
        adversaries_ids = []
        if self.params.fl_number_of_adversaries == 0:
            logger.warning(f'Running vanilla FL, no attack.')
        elif self.params.fl_single_epoch_attack == 0:
            adversaries_ids = random.sample(
                range(self.params.fl_total_participants),
                self.params.fl_number_of_adversaries)
            logger.warning(f'Attacking over multiple epochs with following '
                           f'users compromised: {adversaries_ids}.')
        else:
            logger.warning(f'Attack only on epoch: '
                           f'{self.params.fl_single_epoch_attack} with '
                           f'{self.params.fl_number_of_adversaries} compromised'
                           f' users.')
        print(self.params.fl_single_epoch_attack, adversaries_ids)
    
        return adversaries_ids
    
    def get_model_optimizer(self, model):
        local_model = deepcopy(model)
        local_model = local_model.to(self.params.device)
        optimizer = self.make_optimizer(local_model)
        return local_model, optimizer

    def copy_params(self, global_model, local_model):
        local_state = local_model.state_dict()
        for name, param in global_model.state_dict().items():
            if name in local_state and name not in self.ignored_weights:
                local_state[name].copy_(param)

    def get_fl_update(self, local_model, global_model) -> Dict[str, torch.Tensor]:
        local_update = dict()
        for name, data in local_model.state_dict().items():
            if self.check_ignored_weights(name):
                continue
            local_update[name] = (data - global_model.state_dict()[name])
        return local_update

    def accumulate_weights(self, weight_accumulator, local_update):
        update_norm = self.get_update_norm(local_update)
        for name, value in local_update.items():
            self.dp_clip(value, update_norm)
            weight_accumulator[name].add_(value)

    def update_global_model(self, weight_accumulator, global_model: Module):
        for name, sum_update in weight_accumulator.items():
            if self.check_ignored_weights(name):
                continue
            scale = self.params.fl_eta / self.params.fl_total_participants
            average_update = scale * sum_update
            self.dp_add_noise(average_update)
            model_weight = global_model.state_dict()[name]
            model_weight.add_(average_update)

    def dp_clip(self, local_update_tensor: torch.Tensor, update_norm):
        if self.params.fl_diff_privacy and \
                update_norm > self.params.fl_dp_clip:
            norm_scale = self.params.fl_dp_clip / update_norm
            local_update_tensor.mul_(norm_scale)

    def dp_add_noise(self, sum_update_tensor: torch.Tensor):
        if self.params.fl_diff_privacy:
            noised_layer = torch.FloatTensor(sum_update_tensor.shape)
            noised_layer = noised_layer.to(self.params.device)
            noised_layer.normal_(mean=0, std=self.params.fl_dp_noise)
            sum_update_tensor.add_(noised_layer)

    def get_update_norm(self, local_update):
        squared_sum = 0
        for name, value in local_update.items():
            if self.check_ignored_weights(name):
                continue
            squared_sum += torch.sum(torch.pow(value, 2)).item()
        update_norm = math.sqrt(squared_sum)
        return update_norm

    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True

        return False






# def sample_users_for_round(self, epoch) -> List[FLUser]:
#     single_epoch_attack = self.params.fl_single_epoch_attack
#     multi_epoch_attack = self.params.fl_multi_epoch_attack
#     multi_user_attack = self.params.fl_multi_user_attack

#     # sampled_ids = random.sample(range(self.params.fl_total_participants), self.params.fl_no_models)
#     # random.shuffle(all_range)
#     epoch_id = int((epoch-1) % (self.params.fl_total_participants/self.params.fl_no_models))
#     sampled_ids = list(range(self.params.fl_total_participants))[epoch_id*self.params.fl_no_models: (epoch_id+1)*self.params.fl_no_models]
#     # print(sampled_ids)
#     # exit()
#     sampled_users = []
#     for pos, user_id in enumerate(sampled_ids):
#         A_train_loader = self.fl_A_train_loaders[user_id]
#         compromised = self.check_user_compromised(epoch, pos, user_id)

#         if compromised:
#             if (single_epoch_attack == 0) and (multi_epoch_attack == 0) and (multi_user_attack):
#                 H_train_loader = self.fl_H_train_loaders[self.adversaries.index(user_id)]
#             else:
#                 H_train_loader = self.fl_H_train_loaders[pos]
            
#             # batch_id = 0
#             # for x, y in itr_merge(A_train_loader, H_train_loader):
#             #     input = x if batch_id == 0 else torch.cat((input, x), 0)
#             #     label = y if batch_id == 0 else torch.cat((label, y), 0)
#             #     batch_id = batch_id + 1
#             # concat_dataset = torch.utils.data.TensorDataset(input, label)
#             # A_train_loader = torch.utils.data.DataLoader(concat_dataset, batch_size= self.params.batch_size, shuffle=True, num_workers=0)
            
#         user = FLUser(user_id, compromised=compromised, A_train_loader=A_train_loader, H_train_loader=H_train_loader if compromised else None)
#         sampled_users.append(user)
    
#     return sampled_users