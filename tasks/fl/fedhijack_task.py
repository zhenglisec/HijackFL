import random
from collections import defaultdict

import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# from helper import Helper
from tasks.load_data_model import Data_Model_Task
from tasks.fl.fl_task import FederatedLearningTask
# from models.attack_transformer import UnetGenerator, converter_2, converter_4, converter_6, converter_8

# def build_converter(converter_layer): # 'converter_2, converter_4, converter_6, converter_8'
#     if converter_layer == 'converter_2':
#         Transformer = converter_2()
#     elif converter_layer == 'converter_4':
#         Transformer = converter_4()
#     elif converter_layer == 'converter_6':
#         Transformer = converter_6()
#     elif converter_layer == 'converter_8':
#         Transformer = converter_8()
#     return Transformer

class FedHijackTask(FederatedLearningTask, Data_Model_Task):
    def load_data(self) -> None:
        self.params.build_task = self.params.A_dataset
        A_train_dataset, A_test_dataset = self.build_dataset(self.params.A_dataset)
        Target_Num_Classes = self.A_classes
        ## A dataset
        all_range = list(range(len(A_train_dataset)))
        random.shuffle(all_range)
        self.fl_A_train_loaders = [self.get_train_equally(all_range, pos, A_train_dataset)
                            for pos in range(self.params.fl_total_participants)]
        #  = A_train_loaders
        self.A_test_loader = DataLoader(A_test_dataset, batch_size=self.params.test_batch_size, shuffle=False, num_workers=1)

        ## H dataset
        self.H_transformers = []
        self.fl_H_tasks_train_loaders = []
        self.fl_H_tasks_test_loaders = []
        for H_dataset, H_dataset_size in zip(self.params.H_dataset, self.params.H_dataset_size):
            if H_dataset in ['None', 'none', '']:
                break
            H_dataset_size = int(H_dataset_size)
            # self.params.build_task = H_dataset
            H_train_dataset, H_test_dataset = self.build_dataset(H_dataset, mode='convert')
            self.A_classes = Target_Num_Classes

            all_range = list(range(len(H_train_dataset)))
            random.shuffle(all_range)
            if self.params.hijack_same_subdata:
                sub_indices = all_range[0: H_dataset_size]
                H_train_loaders = [DataLoader(H_train_dataset, batch_size=self.params.batch_size, shuffle=False, sampler=SubsetRandomSampler(sub_indices)) for pos in range(self.params.fl_number_of_adversaries)]            
            else:
                H_train_loaders = [self.get_train_equally(all_range, pos, H_train_dataset, H_dataset_size) for pos in range(self.params.fl_number_of_adversaries)]
            
            # self.H_transformers.append((build_converter(self.params.converter_layer)))
            # from main import check_dataloader
            # check_dataloader(H_train_loaders[0], H_dataset)

            self.fl_H_tasks_train_loaders.append(H_train_loaders)
            self.fl_H_tasks_test_loaders.append(DataLoader(H_test_dataset, batch_size=self.params.test_batch_size, shuffle=False, num_workers=1))

            # self.fl_H_tasks_test_loaders.append(DataLoader(H_test_dataset, batch_size=1 if self.params.hijack_train_setting == 'relabling' else self.params.test_batch_size, shuffle=False, num_workers=2))


        # self.H_data_exits = self.load_H_data()

        # if self.params.H_dataset in ['mnist', 'cifar10', 'cifar100', 'tinyimagenet']:
        #     ## H dataset
        #     all_range = list(range(len(self.H_train_dataset)))
        #     random.shuffle(all_range)
        #     if self.params.hijack_same_subdata:
        #         sub_indices = all_range[0: self.params.H_dataset_size]
        #         H_train_loaders = [DataLoader(self.H_train_dataset, batch_size=self.params.batch_size, sampler=SubsetRandomSampler(sub_indices))
        #                         for pos in range(self.params.fl_number_of_adversaries)]            
        #     else:
        #         H_train_loaders = [self.get_H_train_equally(all_range, pos)
        #                         for pos in range(self.params.fl_number_of_adversaries)]
        #     self.fl_H_train_loaders = H_train_loaders
        # return

    def get_train_equally(self, all_range, model_no, train_dataset, H_dataset_size=None):
        """
        This method equally splits the dataset.
        :param all_range:
        :param model_no:
        :return:
        """

        
        if H_dataset_size is not None:
            data_len = int(len(train_dataset) / self.params.fl_number_of_adversaries)
            sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
            sub_indices = sub_indices[0: H_dataset_size]
            batch_size = self.params.batch_size #1
        else:
            data_len = int(len(train_dataset) / self.params.fl_total_participants)
            sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
            batch_size = self.params.batch_size
        
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size, shuffle=False, 
                                  sampler=SubsetRandomSampler(sub_indices))
        return train_loader

#########################
#########################
#########################
#########################

    # def load_data_old_OK__(self,) -> None:
    #     self.load_A_data()
    #     self.H_data_exits = self.load_H_data()

    #     ## A dataset
    #     if self.params.fl_sample_dirichlet:
    #         # sample indices for participants using Dirichlet distribution
    #         indices_per_participant = self.sample_dirichlet_train_data(
    #             self.params.fl_total_participants,
    #             alpha=self.params.fl_dirichlet_alpha)
    #         train_loaders = [(pos, self.get_train(indices)) for pos, indices in
    #                          indices_per_participant.items()]
    #     else:
    #         # sample indices for participants that are equally
    #         # split to 500 images per participant
    #         all_range = list(range(len(self.A_train_dataset)))
    #         random.shuffle(all_range)
    #         A_train_loaders = [self.get_A_train_equally(all_range, pos)
    #                          for pos in range(self.params.fl_total_participants)]
    #     self.fl_A_train_loaders = A_train_loaders

    #     if self.params.H_dataset in ['mnist', 'cifar10', 'cifar100', 'tinyimagenet']:
    #         ## H dataset
    #         all_range = list(range(len(self.H_train_dataset)))
    #         random.shuffle(all_range)
    #         if self.params.hijack_same_subdata:
    #             sub_indices = all_range[0: self.params.H_dataset_size]
    #             H_train_loaders = [DataLoader(self.H_train_dataset, batch_size=self.params.batch_size, sampler=SubsetRandomSampler(sub_indices))
    #                             for pos in range(self.params.fl_number_of_adversaries)]            
    #         else:
    #             H_train_loaders = [self.get_H_train_equally(all_range, pos)
    #                             for pos in range(self.params.fl_number_of_adversaries)]
    #         self.fl_H_train_loaders = H_train_loaders
    #     return

    # def sample_dirichlet_train_data(self, no_participants, A_train_dataset, alpha=0.9):
    #     """
    #         Input: Number of participants and alpha (param for distribution)
    #         Output: A list of indices denoting data in CIFAR training set.
    #         Requires: cifar_classes, a preprocessed class-indices dictionary.
    #         Sample Method: take a uniformly sampled 10-dimension vector as
    #         parameters for
    #         dirichlet distribution to sample number of images in each class.
    #     """

    #     cifar_classes = {}
    #     for ind, x in enumerate(A_train_dataset):
    #         _, label = x
    #         if ind in self.params.poison_images or \
    #                 ind in self.params.poison_images_test:
    #             continue
    #         if label in cifar_classes:
    #             cifar_classes[label].append(ind)
    #         else:
    #             cifar_classes[label] = [ind]
    #     class_size = len(cifar_classes[0])
    #     per_participant_list = defaultdict(list)
    #     no_classes = len(cifar_classes.keys())

    #     for n in range(no_classes):
    #         random.shuffle(cifar_classes[n])
    #         sampled_probabilities = class_size * np.random.dirichlet(
    #             np.array(no_participants * [alpha]))
    #         for user in range(no_participants):
    #             no_imgs = int(round(sampled_probabilities[user]))
    #             sampled_list = cifar_classes[n][
    #                            :min(len(cifar_classes[n]), no_imgs)]
    #             per_participant_list[user].extend(sampled_list)
    #             cifar_classes[n] = cifar_classes[n][
    #                                min(len(cifar_classes[n]), no_imgs):]

    #     return per_participant_list

    # def get_train(self, indices, A_train_dataset):
    #     """
    #     This method is used along with Dirichlet distribution
    #     :param indices:
    #     :return:
    #     """
    #     train_loader = DataLoader(A_train_dataset,
    #                               batch_size=self.params.batch_size,
    #                               sampler=SubsetRandomSampler(
    #                                   indices))
    #     return train_loader

    # def get_A_train_equally(self, all_range, model_no, A_train_dataset):
    #     """
    #     This method equally splits the dataset.
    #     :param all_range:
    #     :param model_no:
    #     :return:
    #     """

    #     data_len = int(len(A_train_dataset) / self.params.fl_total_participants)
    #     sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
    #     A_train_loader = DataLoader(A_train_dataset,
    #                               batch_size=self.params.batch_size,
    #                               sampler=SubsetRandomSampler(
    #                                   sub_indices))
    #     return A_train_loader

    # def get_H_train_equally(self, all_range, model_no, H_train_dataset):
    #     """
    #     This method equally splits the dataset.
    #     :param all_range:
    #     :param model_no:
    #     :return:
    #     """

    #     data_len = int(len(H_train_dataset) / self.params.fl_number_of_adversaries)
    #     sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
    #     sub_indices = sub_indices[0: self.params.H_dataset_size]
    #     H_train_loader = DataLoader(H_train_dataset,
    #                               batch_size=self.params.batch_size,
    #                               sampler=SubsetRandomSampler(
    #                                   sub_indices))         
    #     return H_train_loader
