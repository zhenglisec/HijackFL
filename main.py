import argparse
from ast import Global
from email.policy import default
from enum import Flag
import shutil
from datetime import datetime
from unittest import TestLoader
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
# torch.set_num_threads(16)
import yaml
from prompt_toolkit import prompt

from helper import Helper
from utils.utils import *
from attack import trainloader_in_FL, testloader_in_FL
from models.converter import UnetGenerator, converter_2, converter_4, converter_6, converter_8, converter_10
logger = logging.getLogger('logger')

def build_converter(hlpr: Helper): 
    if hlpr.params.converter_layer == 'converter_2':
        Converter = converter_2()
    elif hlpr.params.converter_layer == 'converter_4':
        Converter = converter_4()
    elif hlpr.params.converter_layer == 'converter_6':
        Converter = converter_6()
    elif hlpr.params.converter_layer == 'converter_8':
        Converter = converter_8()
    elif hlpr.params.converter_layer == 'converter_10':
        Converter = converter_10()
    return Converter
    
def train(hlpr: Helper, epoch, local_epoch, model, optimizer, user):
    criterion = hlpr.task.criterion
    train_loader, H_train_loader = trainloader_in_FL(user, hlpr, local_epoch, model, Converter)
    total_batches = len(train_loader)

    if (hlpr.params.hijack_train_setting in ['relabling']) and user.compromised:
        TRAIN_FEED = zip(train_loader, H_train_loader)
        if hlpr.params.converted_data == 'joining':
            attack = True
        elif hlpr.params.converted_data == 'no_converter':
            attack = True
        # elif hlpr.params.converted_data == 'joining_delay' :
        #     attack = False
        elif hlpr.params.converted_data in ['not_joining']:
            attack = False
    else: # hlpr.params.hijack_train_setting in ['normal', 'naive']
        TRAIN_FEED = zip(train_loader, train_loader)
        attack = False

    model.train()
    for i, (clean_data, hidden_data) in enumerate(TRAIN_FEED):
        clean_batch = hlpr.task.get_batch(i, clean_data)
        hidden_batch = hlpr.task.get_batch(i, hidden_data)
        model.zero_grad()
        loss = hlpr.attack.compute_hijack_loss(model, criterion, clean_batch, hidden_batch, attack)
        loss.backward() #retain_graph=True
        optimizer.step()
        
        hlpr.report_training_losses_scales(i, epoch, local_epoch, total_batches, user)
        if i == hlpr.params.max_batch_id:
            break

def test(hlpr: Helper, epoch, extra_info=None, hijack=False, local_model=None, H_test_loaders=None):
    hlpr.task.reset_metrics()
    A_test_loaders, H_test_loaders = testloader_in_FL(hlpr, hijack, extra_info, Converter, H_test_loaders=H_test_loaders)
    if not hijack:
        test_loaders = A_test_loaders
    elif hijack:
        test_loaders = H_test_loaders

    if local_model is not None:
        model = local_model
    else:
        model = hlpr.task.model

    model.eval()
    with torch.no_grad():
        for loader_idx, test_loader in enumerate(test_loaders):
            for i, data in enumerate(test_loader):
                batch = hlpr.task.get_batch(i, data)
                outputs = model(batch.inputs)

                if hijack:
                    outputs = outputs[:, 0:10]

                hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
            metric = hlpr.task.report_metrics(epoch,
                                    prefix=f'Hijacking {str(loader_idx if hijack else hijack):5s}. Epoch: ',
                                    logger=hlpr.logger)
    return metric, H_test_loaders

def run_fl_round(hlpr, epoch, extra_info):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model

    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()

    for user in round_participants:   
        if hlpr.params.hijack_train_setting == 'normal':
                user.compromised = False
        if user.compromised:
            if hlpr.params.hijack_train_setting == 'naive':
                pass
            elif hlpr.params.hijack_train_setting == 'relabling':
                if hlpr.params.mapping_func_reset == 1:
                    extra_info = []
                if len(extra_info) > 0:
                    user.mapping_func = extra_info
            
        if epoch < hlpr.params.attacking_start_epoch and (hlpr.params.fl_multi_epoch_attack > 0 or hlpr.params.fl_multi_user_attack) and hlpr.params.fl_single_epoch_attack==0:
                user.compromised = False
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model, hijack=user.compromised)
        
        local_epochs =  hlpr.params.hijack_local_epochs if user.compromised else hlpr.params.fl_local_epochs

        for local_epoch in range(local_epochs):
            train(hlpr, epoch, local_epoch, local_model, optimizer, user)

        local_update = hlpr.task.get_fl_update(local_model, global_model)

        if user.compromised:
            if hlpr.params.hijack_train_setting == 'relabling':
                if len(extra_info) < len(hlpr.params.H_dataset):
                    for mappfuc in user.mapping_func:
                        extra_info.append(mappfuc)
                        
        hlpr.task.accumulate_weights(weight_accumulator, local_update)
    hlpr.task.update_global_model(weight_accumulator, global_model)
    return extra_info

def fl_run(hlpr: Helper):
    global Converter
    extra_info = []
    H_test_loaders = []
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        extra_info = run_fl_round(hlpr, epoch, extra_info)
        
        metric, H_test_loaders = test(hlpr, epoch, extra_info, hijack=False, H_test_loaders=H_test_loaders)
        if hlpr.params.fl_number_of_adversaries > 0:
            _, H_test_loaders = test(hlpr, epoch, extra_info, hijack=True, H_test_loaders=H_test_loaders)
        hlpr.save_model(hlpr.task.model, epoch, metric)
        if hlpr.task.scheduler is not None:
            hlpr.task.scheduler.step(epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', default='configs/fedml_hijack.yaml')
    parser.add_argument('--model_arch', default='resnet_34', help='resnet_34, vgg_16, mobilenet_v2, wideresnet')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--A_dataset', default='food101', help='mnist, qmnist, kmnist, fashion_mnist, emnist, cifar10, cifar100, tinyimagenet, celeba')
    parser.add_argument('--H_dataset', nargs='+', default=['mnist'], help='mnist, emnist, kmnist, fashion_mnist, cifar10, cifar100, tinyimagenet, celebA')
    parser.add_argument('--H_dataset_size', nargs='+', default=[50000], help='1000, 3000, 5000')
        
    parser.add_argument('--fl_local_epochs', default=1, type=int, help='2')
    parser.add_argument('--fl_number_of_adversaries', default=1, type=int, help='0')
    parser.add_argument('--fl_single_epoch_attack', default=0, type=int, help='0')
    parser.add_argument('--fl_multi_epoch_attack', default=1, type=int, help='0') 
    parser.add_argument('--attacking_start_epoch', default=-1, type=int, help='default=1, only work under relabling') 
    parser.add_argument('--fl_multi_user_attack', default='False', type=str, help='False')
    
    ################
    parser.add_argument('--hijack_same_subdata', default=False, type=bool, help='Default False.  True: each attacker controls the same subdata; False: each attacker control different subdata')
    parser.add_argument('--hijack_local_epochs', default=5, type=int)
    parser.add_argument('--hijack_train_setting', default='relabling', help='normal, naive, relabling')
    
    parser.add_argument('--hijack_lr', default=0.05, type=float)
    parser.add_argument('--loss_balance', default='fixed', type=str, help='fixed and MGDA and mixed')
    parser.add_argument('--mgda_normalize', default='loss+', type=str, help='l2 and loss and loss+')
    parser.add_argument('--fixed_scales_hijack', default=0.33, type=float, help='{normal: 1, hijack: 0.33}')

    parser.add_argument('--converted_data', default='not_joining', help='nothing, not_joining, joining')
    parser.add_argument('--converter_layer', default='converter_8', help='converter_2, converter_4, converter_6, converter_8')
    parser.add_argument('--converter_epochs', default=100, type=int)
    parser.add_argument('--converter_multi_updating', default='False', type=str, help='True or False')
    parser.add_argument('--hijack_same_transformer', default='False', type=str, help='Default True. Whether same across different tasks')
    parser.add_argument('--greedy_brute', default='greedy', type=str, help='greedy or brute')
    parser.add_argument('--mapping_func_reset', default='True', type=str, help='True or False')
    # parser.add_argument('--hijack_momentum', default=0.9, type=float)
    # parser.add_argument('--delay_attack', default=False, type=bool, help='True or False')
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['model_arch'] = args.model_arch
    params['epochs'] = int(args.epochs)
    params['A_dataset'] = args.A_dataset
    params['H_dataset'] = args.H_dataset
  
    if args.H_dataset_size[0] == '0000':
        params['H_dataset_size'] = [int(2000/args.fl_number_of_adversaries)] if args.H_dataset[0] in ['mnist'] else [int(50000/args.fl_number_of_adversaries)]
    else:
        params['H_dataset_size'] = args.H_dataset_size #[int(args.H_dataset_size[0])]        
    
    params['fl_local_epochs'] = args.fl_local_epochs
    params['fl_number_of_adversaries'] = args.fl_number_of_adversaries
    params['fl_single_epoch_attack'] = args.fl_single_epoch_attack
    params['fl_multi_epoch_attack'] = args.fl_multi_epoch_attack
    params['attacking_start_epoch'] = args.attacking_start_epoch
    params['fl_multi_user_attack'] = True if args.fl_multi_user_attack  == 'True' else False

    params['hijack_same_subdata'] = args.hijack_same_subdata
    # params['hijack_same_task'] = args.hijack_same_task
    params['hijack_local_epochs'] = args.hijack_local_epochs
    params['hijack_train_setting'] = args.hijack_train_setting
    params['hijack_lr'] = args.hijack_lr
    params['loss_balance'] = args.loss_balance
    params['mgda_normalize'] = args.mgda_normalize

    if args.fixed_scales_hijack != 1.0:
        pass
    elif args.A_dataset in ['cifar100', 'place40'] and args.H_dataset[0] in ['cifar10']:
        args.fixed_scales_hijack = 0.1
    elif args.A_dataset == 'tinyimagenet100' and args.H_dataset[0] in ['cifar10']:
        args.fixed_scales_hijack = 0.05
    else:
        args.fixed_scales_hijack = 0.2
    
    params['fixed_scales'] = {'normal': 1 - args.fixed_scales_hijack, 'hijack': args.fixed_scales_hijack} 
    params['converted_data'] = args.converted_data
    params['converter_layer'] = args.converter_layer
    params['converter_epochs'] = args.converter_epochs
    params['greedy_brute'] = args.greedy_brute
    params['mapping_func_reset'] = True if args.mapping_func_reset == 'True' else False
    params['converter_multi_updating'] = True if args.converter_multi_updating == 'True' else False

    # params['delay_attack'] = args.delay_attack 

    helper = Helper(params)
    logger.warning(create_table(params))

    try:
        # if helper.params.fl:
        Converter = build_converter(helper).to(helper.params.device)
        fl_run(helper)
        # else:
        #     run(helper)
    except (KeyboardInterrupt):
        if helper.params.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.params.folder_path}")
                shutil.rmtree(helper.params.folder_path)
                if helper.params.tb:
                    shutil.rmtree(f'runs/{args.name}')
            else:
                logger.error(f"Aborted training. "
                             f"Results: {helper.params.folder_path}. "
                             f"TB graph: {args.name}")
        else:
            logger.error(f"Aborted training. No output generated.")
# saved_models