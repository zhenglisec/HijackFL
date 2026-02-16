from runx.logx import logx
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.nn import Module
from typing import List, Any, Dict
import math
from utils import load_model, load_dataset, dataset_split, MyDataset
from torch.nn import Module
from torch.utils.data import Subset, DataLoader, ConcatDataset
from copy import deepcopy
from art.estimators.classification import PyTorchClassifier
from art.utils import compute_success
from art.attacks.evasion.targeted_universal_perturbation import TargetedUniversalPerturbation
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader
from torch.nn import BCELoss as BCE

class FederatedLearning:
    def __init__(self, args):
        self.args = args
        self.number_clients_total = args.number_clients_total

        self.build_datasets()
        self.model = load_model(args.model, self.args.p_num_classes)
        self.model.to(args.device)

        self.clients = []
        self.build_clients()

        self.ignored_weights = ['num_batches_tracked']
        self.Noises = None

    def remove_samples(self,dataset, num_classes):
        """
        Remove samples from the dataset with a specific label.

        Args:
            dataset: Dataset containing samples and labels.
            label_to_remove (int): Label to remove from the dataset.

        Returns:
            filtered_dataset: Dataset with samples removed.
        """
        filtered_samples = []
        filtered_labels = []

        for sample, label in dataset:
            if label < int(num_classes):
                filtered_samples.append(sample)
                filtered_labels.append(label)

        filtered_dataset = list(zip(filtered_samples, filtered_labels))
        return filtered_dataset

    def build_datasets(self):
        self.p_trainset, self.p_testset, p_num_classes = load_dataset(self.args, self.args.p_dataset)
        # self.args.p_num_classes = p_num_classes
        if self.args.p_num_classes<p_num_classes:
            self.p_trainset = self.remove_samples(self.p_trainset, self.args.p_num_classes)
            self.p_testset = self.remove_samples(self.p_testset, self.args.p_num_classes)
        self.p_testloader = DataLoader(self.p_testset, batch_size=self.args.batchsize, shuffle=False, num_workers=2) 
        if self.args.number_attackers_of_round > 0:
            h_trainset, h_testset, h_num_classes = load_dataset(self.args, self.args.h_dataset)
            self.h_trainset = self.remove_samples(h_trainset,self.args.h_num_classes)
            self.h_testset = self.remove_samples(h_testset,self.args.h_num_classes)
            self.h_testloader = DataLoader(self.h_testset, batch_size=self.args.batchsize, shuffle=False, num_workers=2) 
            # self.args.h_num_classes = h_num_classes
            
        self.client_dataset_size = int(len(self.p_trainset)/self.args.number_clients_total)

        if self.client_dataset_size * self.args.number_clients_total == len(self.p_trainset):
            self.p_dataset_list = dataset_split(self.p_trainset, self.args.number_clients_total * [self.client_dataset_size])
        elif self.client_dataset_size * self.args.number_clients_total < len(self.p_trainset):
            remain_size = len(self.p_trainset) - self.client_dataset_size * self.args.number_clients_total
            split_list:list = self.args.number_clients_total * [self.client_dataset_size]
            split_list.append(remain_size)
            self.p_dataset_list = dataset_split(self.p_trainset, split_list)
            
    def build_clients(self):
        for client_idx in range(self.number_clients_total):
            if client_idx < self.args.number_attackers_of_round:
                cur_client = Attacker_Client(self.args, client_idx, self.p_dataset_list[client_idx], self.p_testset, self.h_trainset, self.h_testset)
            else:
                cur_client = Normal_Client(self.args, client_idx, self.p_dataset_list[client_idx], self.p_testset)
            self.clients.append(cur_client)

    def run(self):
        for round_idx in range(self.args.rounds):
            logx.msg(f'\n====================== Federated Learning Round {round_idx} ====================')
            weight_accumulator = self.get_empty_accumulator()
            for cur_client_idx in range(self.args.number_clients_of_round):
                client_idx = (round_idx * self.args.number_clients_of_round + cur_client_idx) % self.args.number_clients_total
                cur_client = self.clients[client_idx]
                self.copy_params(self.model, cur_client.model)
                cur_client.run(round_idx)
                local_update = cur_client.get_fl_update(self.model)
                self.accumulate_weights(weight_accumulator, local_update)
                if cur_client.UAP:
                    self.Noises = cur_client.Noises
            self.update_global_model(weight_accumulator, self.model)

            self.eval(round_idx, 'Primary')
            if self.args.number_attackers_of_round > 0: 
                # self.eval(round_idx, 'Hijacking', self.Noises)  
                if round_idx in self.args.attack_rounds:
                    self.eval(round_idx, 'Hijacking', self.Noises)  
                else:
                    self.eval(round_idx, 'Hijacking') 

            if (round_idx + 1) % 10 == 0:
                save_path = self.args.logdir + f'/{round_idx}.pth'
                torch.save(self.model.state_dict(), save_path)
                print(f'Model weights saved at epoch {round_idx}')

    def eval(self, round_idx, mode, Noises=None):
        if mode == 'Primary':
            self.testloader = self.p_testloader
        elif mode == 'Hijacking':
            self.testloader = self.h_testloader

        self.model.eval()
        test_loss = 0
        correct = 0

        if Noises is None:
            with torch.no_grad():
                for data, target in  self.testloader:
                    if data.shape[1] == 1:
                        data = data.repeat(1, 3, 1, 1)
                    data, target = data.to(self.args.device), target.to(self.args.device)
                    output = self.model(data)# 此处应该要裁减
                    test_loss += F.cross_entropy(output, target).item()
                    if mode == 'Hijacking':
                        pred = output[:,:int(self.args.h_num_classes)]
                        pred = pred.max(1, keepdim=True)[1]
                        correct += pred.eq(target.view_as(pred)).sum().item()
                    else:
                        pred = output.max(1, keepdim=True)[1]
                        correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(self.testloader.dataset)
            accuracy = 100. * correct / len(self.testloader.dataset)
            logx.msg('Round: {}, Global Model {} Test Average Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                round_idx, mode, test_loss, correct, len(self.testloader.dataset), accuracy))
        elif Noises is not None: # add UAP noise
            print('Evaluation adding UAP noise')
        
            loss_pred_count = 0
            prob_pred_count = 0
            logit_pred_count = 0
            max_diff_count = 0
            with torch.no_grad():
                for batch_idx, (data, target) in  enumerate(self.testloader):
                    if data.shape[1] == 1:
                        data = data.repeat(1, 3, 1, 1)
                    data, target = data.to(self.args.device), target.to(self.args.device)

                    for each_idx in range(data.shape[0]):
                        cur_data = data[each_idx : each_idx+1]
                        cur_target = target[each_idx : each_idx+1].item()
                        for noise_idx in range(self.args.h_num_classes):

                            W_program = Noises[noise_idx].to(self.args.device)
                            data_noise = W_program.apply(cur_data)
                            data_each_idx = data_noise if noise_idx == 0 else torch.cat((data_each_idx, data_noise), dim=0)
                 
                        output = self.model(data_each_idx)
                        pred = output.max(1, keepdim=True)[1].view(-1)

                        loss = F.cross_entropy(output, pred, reduction='none')
                        loss_pred = loss.min(0, keepdim=True)[1].item()

                        logit = output.max(1, keepdim=True)[0].view(-1)
                        logit_pred = logit.max(0, keepdim=True)[1].item()

                        prob = F.softmax(output, dim=1).max(1, keepdim=True)[0].view(-1)
                        prob_pred = prob.max(0, keepdim=True)[1].item()

                        #####################################
                        output_probs = F.softmax(output, dim=1)
                        num_samples = output_probs.size(0)

                        # 初始化存储差值的张量
                        diff_values = torch.zeros(output_probs.size(0), dtype=output_probs.dtype).to(self.args.device)

                        # 计算每个样本中指定class和最后一位class的差值
                        for i in range(num_samples):
                            diff_values[i] += output_probs[i, i] - output_probs[i, -1]

                        # 找到差值最大的样本的索引
                        max_diff_index = torch.argmax(diff_values).item()
                        #####################################

                        if loss_pred == cur_target:
                            loss_pred_count += 1
                        if prob_pred == cur_target:
                            prob_pred_count += 1
                        if logit_pred == cur_target:
                            logit_pred_count += 1   
                        if max_diff_index == cur_target:
                            max_diff_count += 1
            loss_accuracy = 100. * loss_pred_count / len(self.testloader.dataset)
            prob_accuracy = 100  * prob_pred_count / len(self.testloader.dataset)
            logit_accuracy = 100  * logit_pred_count / len(self.testloader.dataset)
            diff_accuracy = 100  * max_diff_count / len(self.testloader.dataset)
            logx.msg('Round: {}, Global Model {} Test, Loss-based Accuracy: {}/{} ({:.2f}%), Prob-based Accuracy: {}/{} ({:.2f}%), Logit-based Accuracy: {}/{} ({:.2f}%), Diff-based Accuracy: {}/{} ({:.2f}%)'.format(
                round_idx, 
                mode, loss_pred_count, len(self.testloader.dataset), loss_accuracy, 
                prob_pred_count, len(self.testloader.dataset), prob_accuracy,
                logit_pred_count, len(self.testloader.dataset), logit_accuracy,
                max_diff_count, len(self.testloader.dataset), diff_accuracy))
            
    def copy_params(self, global_model, local_model):
        local_state = local_model.state_dict()
        for name, param in global_model.state_dict().items():
            if name in local_state:
                local_state[name].copy_(param)

    def accumulate_weights(self, weight_accumulator, local_update):
        # update_norm = self.get_update_norm(local_update)
        for name, value in local_update.items():
            # self.dp_clip(value, update_norm)
            weight_accumulator[name].add_(value)

    def update_global_model(self, weight_accumulator, global_model: Module):
        for name, sum_update in weight_accumulator.items():
            if self.check_ignored_weights(name):
                continue
            scale = self.args.eta / self.args.number_clients_of_round
            average_update = scale * sum_update
            # self.dp_add_noise(average_update)
            model_weight:torch.Tensor = global_model.state_dict()[name]
            model_weight.add_(average_update)

    def get_empty_accumulator(self):
        weight_accumulator = dict()
        for name, data in self.model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)
        return weight_accumulator
    
    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True
        return False
    
class Normal_Client:
    def __init__(self, args, client_idx, trainset, testset):
        self.args = args
        self.client_idx = client_idx
        self.model = load_model(self.args.model, self.args.p_num_classes)
        self.model.to(args.device)

        self.trainloader = DataLoader(trainset, batch_size=self.args.batchsize, shuffle=False, num_workers=2) 
        self.testloader = DataLoader(testset, batch_size=self.args.batchsize, shuffle=False, num_workers=2)  

        self.ignored_weights = ['num_batches_tracked']
        self.UAP = False
       
    def run(self, round_idx):
        optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.args.lr,
                                  weight_decay=self.args.decay,
                                  momentum=self.args.momentum)
        for epoch in range(1, self.args.normal_local_epochs + 1):
            self.train_model(round_idx, epoch, optimizer)
        # self.test_model(round_idx)
    
    def train_model(self, round_idx, epoch, optimizer):
        self.model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(self.trainloader):
            if data.shape[1] == 1:
                data = data.repeat(1, 3, 1, 1)
            data, target = data.to(self.args.device), target.to(self.args.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss /= len(self.trainloader.dataset)
        accuracy = 100. * correct / len(self.trainloader.dataset)
        logx.msg('Round: {}, Normal Client: {}, Epoch: {}, Train Average Loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'.format(
            round_idx,
            self.client_idx,
            epoch,
            train_loss,
            correct, len(self.trainloader.dataset), accuracy))
        # self.test_model(round_idx)
        
    def test_model(self, round_idx):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.testloader:
                if data.shape[1] == 1:
                    data = data.repeat(1, 3, 1, 1)
                data, target = data.to(self.args.device), target.to(self.args.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.testloader.dataset)
        accuracy = 100. * correct / len(self.testloader.dataset)
        logx.msg('Round: {}, Normal Client: {}, Test Average Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            round_idx, self.client_idx, test_loss, correct, len(self.testloader.dataset), accuracy))

    def get_fl_update(self, global_model) -> Dict[str, torch.Tensor]:
        local_update = dict()
        for name, data in self.model.state_dict().items():
            if self.check_ignored_weights(name):
                continue
            local_update[name] = (data - global_model.state_dict()[name])
        return local_update
    
    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True
        return False
    
class DataSet(Dataset):
    def __init__(self,data,label):
        self.data_size=len(data)
        self.data=data
        self.label=label

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, index):
        img,label = self.data[index],self.label[index]
        return img,label
    

class Attacker_Client:
    def __init__(self, args, client_idx, p_trainset, p_testset, h_trainset, h_testset):
        self.args = args
        self.client_idx = client_idx
        self.model = load_model(self.args.model, self.args.p_num_classes)
        self.model.to(args.device)

        self.p_trainset = p_trainset
        self.h_trainset = h_trainset
        self.p_testloader = DataLoader(p_testset, batch_size=self.args.batchsize, shuffle=False, num_workers=2)   
        self.h_testloader = DataLoader(h_testset, batch_size=self.args.batchsize, shuffle=False, num_workers=2)  

        self.ignored_weights = ['num_batches_tracked']
        self.UAP = False
        self.Noises = None

    def run(self, round_idx):
        if self.args.train_mode == 'ahmed':
            self.run_ahmed(round_idx) 
            self.test_model(self.model, round_idx, 'Primary', self.p_testloader)
            self.test_model(self.model, round_idx, 'Hijacking', self.h_testloader)
        elif self.args.train_mode == 'UAP':
            self.UAP = True
            self.run_UAP(round_idx)
            self.test_model(self.p_model, round_idx, 'Primary', self.p_testloader)
            # self.test_model(self.h_model, round_idx, 'Hijacking', self.h_testloader)
        elif self.args.train_mode == 'hidden':
            self.run_hidden(round_idx)
            self.test_model(self.model, round_idx, 'Primary', self.p_testloader)
            self.test_model(self.model, round_idx, 'Hijacking', self.h_testloader)

    def hidden_data_loader(self):
        data = np.load('./encoder_decoder/hidden_data_{}_{}.npz'.format(self.args.p_dataset,self.args.h_dataset))
        img,label = data['data'],data['label']
        dataset=DataSet(img,label)
        dalaloader = DataLoader(dataset,batch_size=self.args.batchsize,shuffle=False,num_workers=2)
        return dalaloader

    def run_ahmed(self, round_idx):
        trainloader = self.build_trainloader(round_idx)
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.attacker_lr, weight_decay=self.args.decay, momentum=self.args.momentum)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4], gamma=0.1)
        for epoch in range(1, self.args.attacker_local_epochs + 1):
            self.train_model(self.model, round_idx, epoch, optimizer, trainloader)
            scheduler.step()

    def run_hidden(self,round_idx):
        trainloader = self.hidden_data_loader()
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.attacker_lr, weight_decay=self.args.decay, momentum=self.args.momentum)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4], gamma=0.1)
        for epoch in range(1, self.args.attacker_local_epochs + 1):
            self.train_model(self.model, round_idx, epoch, optimizer, trainloader)
            scheduler.step()

    def run_UAP(self, round_idx):
        self.p_model = deepcopy(self.model)
        self.p_model = self.p_model.to(self.args.device)
        self.h_model = deepcopy(self.model)
        self.h_model = self.h_model.to(self.args.device)
        p_trainloader, h_trainloader = self.build_trainloader()
        p_optimizer = optim.SGD(self.p_model.parameters(), lr=self.args.attacker_lr, weight_decay=self.args.decay, momentum=self.args.momentum)
        p_scheduler = optim.lr_scheduler.MultiStepLR(p_optimizer, milestones=[2,4], gamma=0.1)
        h_optimizer = optim.SGD(self.h_model.parameters(), lr=self.args.attacker_lr, weight_decay=self.args.decay, momentum=self.args.momentum)
        h_scheduler = optim.lr_scheduler.MultiStepLR(h_optimizer, milestones=[2,4], gamma=0.1)

        for epoch in range(1, self.args.attacker_local_epochs + 1):
            self.train_model(self.p_model, round_idx, epoch, p_optimizer, p_trainloader)
            p_scheduler.step()

        if round_idx not in self.args.attack_rounds:
            return
        
        # h_trainloader = self.dataloader_UAP_noise(h_trainloader)
        self.dataloader_AR_noise(h_trainloader)
 
        # for epoch in range(1, self.args.attacker_local_epochs + 1):
        #     self.train_model(self.h_model, round_idx, epoch, h_optimizer, h_trainloader)
        #     h_scheduler.step()  

    def dataloader_UAP_noise(self, h_trainloader):
        ART_h_classifier = PyTorchClassifier(
                model=self.h_model,
                loss=F.cross_entropy,
                input_shape=(3, self.args.img_size, self.args.img_size),
                clip_values=(0, 1),
                nb_classes=self.args.p_num_classes)
        ART_UAP = TargetedUniversalPerturbation(
                ART_h_classifier,
                attacker='fgsm',
                delta=self.args.UAP_delta,
                attacker_params={'targeted':True, 'eps':self.args.fgsm_eps},
                max_iter=self.args.uap_max_iter,
                eps=self.args.uap_eps,
                norm=2)
        
        Selected_Data, Selected_Target = [], []
        for _ in range(self.args.h_num_classes):
            Selected_Data.append([])
            Selected_Target.append([])

        for batch_idx, (data, target) in enumerate(h_trainloader):
            if data.shape[1] == 1:
                data = data.repeat(1, 3, 1, 1)
            for each_idx in range(target.shape[0]):
                sample = np.array(data[each_idx : each_idx+1])
                target_class = target[each_idx : each_idx+1].item()

                label = np.array(F.one_hot(target[each_idx : each_idx+1], num_classes=self.args.p_num_classes))

                if type(Selected_Data[target_class]) == list:
                    Selected_Data[target_class]= sample
                    Selected_Target[target_class] = label
                else:
                    Selected_Data[target_class] = np.concatenate((Selected_Data[target_class], sample), axis=0)
                    Selected_Target[target_class] = np.concatenate((Selected_Target[target_class], label), axis=0)
        
        self.Noises = []
        for target_class in range(self.args.h_num_classes):
            cur_data, cur_labels = Selected_Data[target_class][:1000], Selected_Target[target_class][:1000]
            logx.msg(f'=================== UAP Noise Generation, Class: {target_class} =================')

            _ = ART_UAP.generate(cur_data, y=cur_labels)
            noise = ART_UAP.noise[0, :]

            if target_class == 0:
                h_trainset_UAP_data  = cur_data + noise
                h_trainset_UAP_targets = np.zeros(cur_labels.shape[0]) + target_class
            else:
                h_trainset_UAP_data = np.concatenate((h_trainset_UAP_data, cur_data + noise), axis=0)
                h_trainset_UAP_targets = np.concatenate((h_trainset_UAP_targets, np.zeros(cur_labels.shape[0]) + target_class), axis=0)
            
            self.Noises.append(noise)

        h_trainset_UAP = torch.utils.data.TensorDataset(
                torch.from_numpy(np.array(h_trainset_UAP_data)).type(torch.float),
                torch.from_numpy(h_trainset_UAP_targets).type(torch.long))
        h_trainloader = DataLoader(h_trainset_UAP, batch_size=self.args.batchsize, shuffle=True, num_workers=2)  
        
        # for batch_idx, (data, targets) in enumerate(self.h_testloader):
        #     for each_idx in range(data.shape[0]):
        #         noise_tensor = torch.from_numpy(self.Noises[targets[each_idx : each_idx+1].item()]).type(torch.float)
        #         cur_data = data[each_idx : each_idx+1] + noise_tensor
        #         h_test_data = cur_data if batch_idx == 0 and each_idx == 0 else torch.cat((h_test_data, cur_data), dim=0)
        #     h_test_targets = targets if batch_idx == 0 else torch.cat((h_test_targets, targets), dim=0)
        # h_testset_UAP = torch.utils.data.TensorDataset(h_test_data, h_test_targets)
        # self.h_testloader = DataLoader(h_testset_UAP, batch_size=self.args.batchsize, shuffle=True, num_workers=2)  

        return h_trainloader

    def dataloader_AR_noise(self, h_trainloader):
        
        # Sort images and labels by class
        sorted_images = [[] for _ in range(10)]  # 10 classes in MNIST
        sorted_labels = [[] for _ in range(10)]

        for images, labels in h_trainloader:
            for image, label in zip(images, labels):
                sorted_images[label].append(image)
                sorted_labels[label].append(label)

        # Create a new dataset with sorted images and labels


        for class_index in range(10):
            sorted_dataset = []
            sorted_dataset.extend(list(zip(sorted_images[class_index], sorted_labels[class_index])))

            # Define a custom DataLoader for the new dataset
            batch_size = 64  # Adjust as needed
            sorted_loader = torch.utils.data.DataLoader(sorted_dataset, batch_size=batch_size, shuffle=True)

            # Now, you can use sorted_loader to train your classification model
            logx.msg(f'=================== AR Noise Generation, Class: {class_index} =================')
            W = Program(self.args, primary_size = (3, self.args.img_size, self.args.img_size), hijack_size = (3, 28, 28)).to(self.args.device)

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, W.parameters()), lr=0.005, betas=(0.5, 0.999))
            # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.96)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
            # Example usage of sorted_loader:
            for images, labels in sorted_loader:
                lr_scheduler.step()

                images = images.to(self.args.device)
                labels = torch.zeros(labels.shape[0], 10).scatter_(1, labels.view(-1,1), 1)
                labels = labels.to(self.args.device)
                print(images.shape)
                print(labels)
                # Your training code here
                print("Batch size:", len(images))
                break

   

            
            
            
            for epoch in range(30):
                train_loss = 0
                correct = 0
                for batch, (data, target) in enumerate(cur_dataloader):
                    if data.shape[1] == 1:
                        data = data.repeat(1, 3, 1, 1)
                    data = data.to(self.args.device)
                    data_adv = W(data)

                    target = target * 0 + target_class
                    target_onehot = torch.zeros(target.shape[0], 10).scatter_(1, target.view(-1,1), 1)
                    target_onehot = target_onehot.to(self.args.device)
                    
                    output_adv = self.h_model(data_adv)
                    output_adv = F.softmax(output_adv, 1)[:, : self.args.h_num_classes]
                    # out = imagenet_label2_mnist_label(output_adv)
                    loss = BCE(output_adv, target_onehot) + (self.args.ar_lamda) * torch.norm(self.get_W(W)) ** 2
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    pred = output_adv.max(1, keepdim=True)[1]
                    correct += pred.eq(target.to(self.args.device).view_as(pred)).sum().item()

                train_loss /= len(cur_dataloader.dataset)
                accuracy = 100. * correct / len(cur_dataloader.dataset)

                logx.msg('AR Noise Epoch: {}, Loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'.format(epoch, loss.item(), correct, len(cur_dataloader.dataset), accuracy))
                lr_scheduler.step()

                # if accuracy >= 95:
                #     break

            self.Noises.append(W)
    
    def dataloader_AR_one_noise(self, h_trainloader):
        selected_size = 1000
        Selected_Data, Selected_Target = [], []
        for _ in range(self.args.h_num_classes):
            Selected_Data.append([])
            Selected_Target.append([])

        for batch_idx, (data, target) in enumerate(h_trainloader):
            if data.shape[1] == 1:
                data = data.repeat(1, 3, 1, 1)
            for each_idx in range(target.shape[0]):
                sample = data[each_idx : each_idx+1]
                target_class = target[each_idx : each_idx+1].item()

                label = target[each_idx : each_idx+1]

                if type(Selected_Data[target_class]) == list:
                    Selected_Data[target_class]= sample
                    Selected_Target[target_class] = label
                elif len(Selected_Data[target_class]) == selected_size:
                    continue
                else:
                    Selected_Data[target_class] = torch.cat((Selected_Data[target_class], sample), dim=0)
                    Selected_Target[target_class] = torch.cat((Selected_Target[target_class], label), dim=0)
        
        self.Noises = []
        self.h_model.eval()

        cur_data, cur_labels = torch.cat(Selected_Data, dim=0), torch.cat(Selected_Target, dim=0)
        cur_dataset = torch.utils.data.TensorDataset(cur_data, cur_labels)
        cur_dataloader = torch.utils.data.DataLoader(cur_dataset, batch_size=100, shuffle=True, num_workers=1)
        # logx.msg(f'=================== AR Noise Generation, Class: {target_class} =================')

        W = Program(mask_shape=(3, self.args.img_size, self.args.img_size)).to(self.args.device)
    
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, W.parameters()), lr=0.005, betas=(0.5, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.96)
        for epoch in range(30):
            train_loss = 0
            correct = 0
            for batch, (data, target) in enumerate(cur_dataloader):
                if data.shape[1] == 1:
                    data = data.repeat(1, 3, 1, 1)
                data = data.to(self.args.device)
                data_adv = W(data)

                # target = target * 0 + target_class
                target = target.to(self.args.device)
                
                output_adv = self.h_model(data_adv)[:, : self.args.h_num_classes]
                loss = F.cross_entropy(output_adv, target) + (self.args.ar_lamda) * torch.norm(self.get_W(W)) ** 2
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pred = output_adv.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

            train_loss /= len(cur_dataloader.dataset)
            accuracy = 100. * correct / len(cur_dataloader.dataset)

            logx.msg('AR Noise Epoch: {}, Average Loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'.format(epoch, train_loss, correct, len(cur_dataloader.dataset), accuracy))
            lr_scheduler.step()

            # if accuracy >= 95:
            #     break

        self.Noises.append(W)
        exit()
    
    def dataloader_AR_noise_with_negative(self, h_trainloader):
        selected_size = 1000
        Selected_Data, Selected_Target = [], []
        for _ in range(self.args.h_num_classes):
            Selected_Data.append([])
            Selected_Target.append([])

        for batch_idx, (data, target) in enumerate(h_trainloader):
            if data.shape[1] == 1:
                data = data.repeat(1, 3, 1, 1)
            for each_idx in range(target.shape[0]):
                sample = data[each_idx : each_idx+1]
                target_class = target[each_idx : each_idx+1].item()

                label = target[each_idx : each_idx+1]

                if type(Selected_Data[target_class]) == list:
                    Selected_Data[target_class]= sample
                    Selected_Target[target_class] = label
                elif len(Selected_Data[target_class]) == selected_size:
                    continue
                else:
                    Selected_Data[target_class] = torch.cat((Selected_Data[target_class], sample), dim=0)
                    Selected_Target[target_class] = torch.cat((Selected_Target[target_class], label), dim=0)
        
        self.Noises = []
        self.h_model.eval()
        for target_class in range(self.args.h_num_classes):
            dataset = Selected_DataSet(Selected_Data, target_class)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, num_workers=1)
            # cur_data, cur_labels = Selected_Data[target_class][:selected_size], Selected_Target[target_class][:selected_size]
            # cur_dataset = torch.utils.data.TensorDataset(cur_data, cur_labels)
            # cur_dataloader = torch.utils.data.DataLoader(cur_dataset, batch_size=100, shuffle=True, num_workers=1)
            logx.msg(f'=================== AR Noise Generation, Class: {target_class} =================')

            W = Program(mask_shape=(3, self.args.img_size, self.args.img_size)).to(self.args.device)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, W.parameters()), lr=0.005, betas=(0.5, 0.999))
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.96)
            # W.train()
            for epoch in range(30):
                train_loss = 0
                train_loss_pos = 0
                train_loss_neg = 0
                correct = 0
                for batch, (data, negative, target) in enumerate(dataloader):
                    if data.shape[1] == 1:
                        data = data.repeat(1, 3, 1, 1)
                    data = data.to(self.args.device)
                    data_adv = W(data)
                    negative = negative.to(self.args.device)
                    negative_adv = W(negative)
                    target = target.to(self.args.device)
                    # target_adv = target * 0 + 9

                    output_adv = self.h_model(data_adv)[:, : self.args.h_num_classes]
                    output_neg_adv = self.h_model(negative_adv)[:, : self.args.h_num_classes]

                    loss_pos = F.cross_entropy(output_adv, target)
                    loss_neg = F.cross_entropy(output_neg_adv, target)

                    loss = loss_pos - (self.args.ar_lamda) * loss_neg
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    train_loss_pos += loss_pos.item()
                    train_loss_neg += loss_neg.item()
                    
                    pred = output_adv.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()

                train_loss /= len(dataloader.dataset)
                train_loss_pos /= len(dataloader.dataset)
                train_loss_neg /= len(dataloader.dataset)
                accuracy = 100. * correct / len(dataloader.dataset)

                logx.msg('AR Noise Epoch: {}, Average Loss: {:.6f}, Positive Loss: {:.6f}, Negative Loss: {:.6f} Accuracy: {}/{} ({:.2f}%)'.format(epoch,
                                                                     train_loss, train_loss_pos, train_loss_neg, correct, len(dataloader.dataset), accuracy))
                lr_scheduler.step()

                # if accuracy >= 95:
                #     break

            self.Noises.append(W.eval())
    
    def get_fl_update(self, global_model) -> Dict[str, torch.Tensor]:
        if self.args.train_mode == 'ahmed':
            local_update = dict()
            for name, data in self.model.state_dict().items():
                if self.check_ignored_weights(name):
                    continue
                local_update[name] = self.args.eugene_scaling * (data - global_model.state_dict()[name])
            return local_update
        

        elif self.args.train_mode == 'UAP':
            p_local_update = dict()
            h_local_update = dict()
            for name, data in self.p_model.state_dict().items():
                if self.check_ignored_weights(name):
                    continue
                p_local_update[name] = (data - global_model.state_dict()[name])

            for name, data in self.h_model.state_dict().items():
                if self.check_ignored_weights(name):
                    continue
                h_local_update[name] = (data - global_model.state_dict()[name])
            
            local_update = dict()
            for name, _ in self.model.state_dict().items():
                if self.check_ignored_weights(name):
                    continue
                local_update[name] = self.args.eugene_scaling * ((1-self.args.flhj_scaling) * p_local_update[name] + self.args.flhj_scaling * h_local_update[name])
            return local_update
        

        elif self.args.train_mode == 'hidden':
            local_update = dict()
            for name, data in self.model.state_dict().items():
                if self.check_ignored_weights(name):
                    continue
                local_update[name] = self.args.eugene_scaling * (data - global_model.state_dict()[name])
                # 怎么更新？？？
            return local_update

        
    def train_model(self, model, round_idx, epoch, optimizer, trainloader):
        model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(trainloader):
            if data.shape[1] == 1:
                data = data.repeat(1, 3, 1, 1)
            data, target = data.to(self.args.device), target.to(self.args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss /= len(trainloader.dataset)
        accuracy = 100. * correct / len(trainloader.dataset)
        logx.msg('Round: {}, Attacker Client: {}, Epoch: {}, Train Average Loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'.format(
            round_idx,
            self.client_idx,
            epoch,
            train_loss,
            correct, len(trainloader.dataset), accuracy))
        
    def test_model(self, model, round_idx, mode, testloader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in testloader:
                if data.shape[1] == 1:
                    data = data.repeat(1, 3, 1, 1)
                data, target = data.to(self.args.device), target.to(self.args.device)
                output = model(data)
                test_loss += F.cross_entropy(output, target).item()
                if mode=='Hijacking':
                    pred = output[:,:int(self.args.h_num_classes)]
                    pred = pred.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
                else:
                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(testloader.dataset)
        accuracy = 100. * correct / len(testloader.dataset)
        logx.msg('Round: {}, Attacker Client: {}, {} Test Average Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            round_idx, self.client_idx, mode, test_loss, correct, len(testloader.dataset), accuracy))
        
    def build_trainloader(self, round_idx=None):
        if self.args.train_mode == 'ahmed':
            if round_idx not in self.args.attack_rounds:
                trainloader = DataLoader(self.p_trainset, batch_size=self.args.batchsize, shuffle=True, num_workers=2)  
                # mixed_dataset = ConcatDataset([self.p_trainset, self.h_trainset])
            else:
                mixed_dataset = ConcatDataset([self.p_trainset, self.h_trainset])
                trainloader = DataLoader(mixed_dataset, batch_size=self.args.batchsize, shuffle=True, num_workers=2)  
            return trainloader
        elif self.args.train_mode == 'UAP':
            p_trainloader = DataLoader(self.p_trainset, batch_size=self.args.batchsize, shuffle=True, num_workers=2)  
            h_trainloader = DataLoader(self.h_trainset, batch_size=self.args.batchsize, shuffle=True, num_workers=2)  
            return p_trainloader, h_trainloader

    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True
        return False
    
    def get_W(self, Program):
        for p in Program.parameters():
            if p.requires_grad:
                return p

class Program(nn.Module):
    def __init__(self, args, primary_size, hijack_size):
        super(Program, self).__init__()
        self.args = args
        self.w1, self.h1 = primary_size[1], primary_size[2]
        self.w2, self.h2 = hijack_size[1], hijack_size[2]
        # create mask M
        M = np.ones((3, self.h1, self.w1), dtype=np.float32)
        c_w, c_h = int(np.ceil(self.w1/2.)), int(np.ceil(self.h1/2.))
        M[:,c_h-self.h2//2:c_h+self.h2, c_w-self.w2//2:c_w+self.w2//2] = 0
        self.M = torch.from_numpy(M).to(args.device)
        # self.M = Parameter(M, requires_grad=True)

        # Learnable parameter W
        W = torch.rand(M.shape)
        self.W = Parameter(W, requires_grad=True) 



        # self.W = Parameter(torch.randn(mask_shape), requires_grad=True)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(3),
        #     nn.Sigmoid(),
        # )
        # self.transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        #         ])
        # self.transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 

    def forward(self, data):
        batch_size = data.shape[0]
        X = torch.zeros((batch_size, 3, self.h1, self.w1)).to(self.args.device)
        X[:,:,(self.h1-self.h2)//2:(self.h1+self.h2)//2, (self.w1-self.w2)//2:(self.w1+self.w2)//2] = data
        # X = torch.from_numpy(X).to(self.args.device)

        P = torch.sigmoid(self.W * self.M)
        X_adv = X + P # range [0, 1]

        # w = F.sigmoid(self.W)
        # w = F.tanh(w)
        # data_adv = data + self.W
        # data_adv = F.sigmoid(data_adv)
        # data_adv = torch.clamp(data_adv, -1, 1)
        # data_adv = self.conv(data_adv)
        # data_adv = self.transform(data_adv.view(-1, 3, args.img_size, args.img_size))
        # data_adv = data_adv.view(-1, 3, args.img_size, args.img_size)
        return X_adv
    
    def apply(self, data):
        batch_size = data.shape[0]
        X = torch.zeros(batch_size, 3, self.h1, self.w1).to(self.args.device)
        X[:,:,(self.h1-self.h2)//2:(self.h1+self.h2)//2, (self.w1-self.w2)//2:(self.w1+self.w2)//2] = data
        P = torch.sigmoid(self.W * self.M)
        X_adv = X + P # range [0, 1]
        # w = F.sigmoid(self.W)
        # data_adv = data + self.W
        # data_adv = F.sigmoid(data_adv)
        # data_adv = torch.clamp(data_adv, -1, 1)
        # data_adv = self.conv(data_adv)
        # data_adv = self.transform(data_adv.view(-1, 3, args.img_size, args.img_size))
        # data_adv = data_adv.view(-1, 3, args.img_size, args.img_size)
        return X_adv
    
class Selected_DataSet(torch.utils.data.Dataset):
    # 80 * 750 = 60,000
    def __init__(self, Selected_Data, class_index):
        self.Selected_Data = Selected_Data
        self.class_index = class_index
        self.data = Selected_Data[class_index]
        self.num_class = len(Selected_Data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        while True:
            random_class = random.randint(0, self.num_class-1)
            if random_class != self.class_index:
                break
        # print(random_class, index, len(self.Selected_Data[random_class]))
        random_index = random.randint(0, len(self.Selected_Data[random_class])-1)
        negative = self.Selected_Data[random_class][random_index]

        target = self.class_index
        return data, negative, target
    

class MyDataSet(torch.utils.data.Dataset):
    # 80 * 750 = 60,000
    def __init__(self, Selected_Data, class_index):
        self.class_index = class_index
        self.data = Selected_Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index],self.class_index
