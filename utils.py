from PIL import Image
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps
import os
import shutil
import pandas as pd
from torch._utils import _accumulate
from torch import randperm
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
import torch
import torch.backends.cudnn as cudnn
# import seaborn as sns
import matplotlib.legend
# from datasets import load_dataset
# from models import resnet_34, mobilenet_v2, vgg_16, preact_resnet
from models import densenet121, resnet50, vgg19_bn, googlenet, shufflenetv2, inceptionv4, resnet18, resnet34, resnext50, seresnet50 , mobilenetv2, vgg16_bn
from models.preactresnet import preactresnet18
from torch.utils.data import Dataset, DataLoader

from img2dataset import logos_dataset

# Assuming you have a custom dataset class
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.labels[index]
        sample = torch.from_numny(sample).float()
        return sample, target
    
def filter_dataset(dataset, target_class):
    indices = torch.nonzero(dataset.labels == target_class).squeeze()
    selected_data = dataset.data[indices]
    selected_labels = dataset.labels[indices]

    filtered_dataset = MyDataset(selected_data, selected_labels)
    return filtered_dataset

def list_str_to_int(the_list):
    for idx, item in enumerate(the_list):
        the_list[idx] = int(item)
    return the_list
def load_model(model, num_classes):
    if model == 'resnet18':
        model = resnet18(num_classes)
    elif model == 'resnet50':
        model = resnet50(num_classes)
    elif model =='mobilenetv2':
        model = mobilenetv2(num_classes)
    elif model == 'vgg16':
        model = vgg16_bn(num_classes)
    elif model == 'preactresnet18':
        model = preactresnet18(num_classes)
    elif model == 'vgg19':
        model = vgg19_bn(num_classes)
    elif model == 'shufflenetv2':
        model = shufflenetv2(num_classes)
    else:
        print('Building Model Failed...')
        exit()
    return model
def load_loader(dataset, batchsize):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=2)
    return loader

def load_dataset(args, dataset, mode='p'):
    root_path='./datas/'
    if mode == 'p':
        print('load_dataset...p')
        transform_train = transforms.Compose([
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomRotation(15),
                    Rand_Augment(),
                    transforms.Resize((args.img_size,args.img_size)),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
                    ])
        transform_test = transforms.Compose([
                transforms.Resize((args.img_size,args.img_size)),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    elif mode == 'h':
        print('load_dataset...h')
        transform_train = transforms.Compose([
                    transforms.Resize((args.img_size,args.img_size)),
                    transforms.ToTensor(),
                    ])
        transform_test = transforms.Compose([
                transforms.Resize((args.img_size,args.img_size)),
                transforms.ToTensor(),
                ])
    if dataset == 'mnist':
        trainset = datasets.MNIST(root_path, train=True, download=True, 
                                    transform=transforms.Compose([
                                                transforms.Resize(args.img_size),
                                                transforms.Grayscale(3),
                                                transforms.ToTensor(),
                                                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ]) )
        testset = datasets.MNIST(root_path, train=False, download=True, transform=transforms.Compose([
                                                                    transforms.Resize(args.img_size),
                                                                    # transforms.Grayscale(3),
                                                                    transforms.ToTensor(),
                                                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                                    ]))
        num_classes = 10
    elif dataset == 'cifar10':
        trainset = datasets.CIFAR10(root_path, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root_path, train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset == 'cifar100':
        trainset = datasets.CIFAR100(root_path, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root_path, train=False, download=True, transform=transform_test)
        num_classes = 100
    elif dataset == 'tinyimagenet200':
        train_dir = root_path+'tinyimagenet/my-200/train'
        valid_dir = root_path+'tinyimagenet/my-200/val'
        trainset =  datasets.ImageFolder(train_dir, transform=transform_train)
        testset =  datasets.ImageFolder(valid_dir, transform=transform_test)
        num_classes = 200 
    elif dataset == 'tinyimagenet100':
        train_dir = root_path+'tinyimagenet100/train'
        valid_dir = root_path+'tinyimagenet100/val'
        trainset =  datasets.ImageFolder(train_dir, transform=transform_train)
        testset =  datasets.ImageFolder(valid_dir, transform=transform_test)
        num_classes = 100 
    elif dataset == 'gtsrb':
        trainset = datasets.GTSRB(root_path, split='train', download=True, transform=transform_train)
        testset = datasets.GTSRB(root_path, split='test', download=True, transform=transform_test)
        # train_dir = '/p/scratch/hai_glaze/FedML/dataset/gtsrb/GTSRB/Training'
        # valid_dir = '/p/scratch/hai_glaze/FedML/dataset/gtsrb/GTSRB/Testing'
        # trainset =  datasets.ImageFolder(train_dir, transform=transform_train)
        # testset =  datasets.ImageFolder(valid_dir, transform=transform_test)
        num_classes = 43
    elif dataset == 'stl10':
        trainset = datasets.STL10(
                    root=root_path,
                    split='train',
                    download=True,
                    transform=transform_train)
        testset = datasets.STL10(
                    root=root_path,
                    split='test', 
                    download=True, 
                    transform=transform_test)
        num_classes = 10 
    elif dataset == 'svhn':
        trainset = datasets.SVHN(
                    root=root_path+'svhn',
                    split='train',
                    download=True,
                    transform=transform_train)
        testset = datasets.SVHN(
                    root=root_path+'svhn',
                    split='test', 
                    download=True, 
                    transform=transform_test)
        num_classes = 10 
    elif dataset == 'celeba':
        trainset = datasets.CelebA(
                    root=root_path+'celeba',
                    split='train',
                    target_type='attr',
                    download=True,
                    transform=transform_train)
        testset = datasets.CelebA(
                    root=root_path+'celeba',
                    split='test', 
                    target_type='attr',
                    download=True, 
                    transform=transform_test)
        num_classes = 10 
    elif dataset == 'Caltech101':
        dataset = datasets.Caltech101(
                    root=root_path+'Caltech101',
                    target_type='category',
                    download=True,
                    transform=transform_test)
        fulll_length = len(dataset)
        train_size = 50000
        test_size = fulll_length - train_size
        trainset, testset = dataset_split(dataset, [train_size, test_size])
        num_classes= 101
    elif dataset == 'logos10':
        trainset = logos_dataset(
            split='train', 
            transform=transform_test, 
        )
        testset = logos_dataset(
            split='test', 
            transform=transform_test, 
        )
        num_classes = 10
    elif dataset == 'logos6':
        trainset = logos_dataset(
            split='train', 
            balenced=True, 
            transform=transform_train, 
        )
        testset = logos_dataset(
            split='test', 
            balenced=True, 
            transform=transform_test, 
        )
        num_classes = 6

    return trainset, testset, num_classes

def dataset_split(dataset, lengths):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = list(range(sum(lengths)))
    np.random.seed(1)
    np.random.shuffle(indices)
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]

def fixed_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
def check_args(args):
    if args.train_mode == 'normal' and args.number_attackers_of_round == 0:
        pass
    elif args.train_mode in ['ahmed', 'Fusion', 'UAP','hidden','multi_mask','one_mask'] and args.number_attackers_of_round > 0:
        pass
    else:
        print('===========args.train_mode MISMATCH args.number_attackers_of_round!!!===============')
        exit()

    # if args.p_dataset in ['cifar10', 'tinyimagenet100']:
    #     args.eta = 10
    # elif args.p_dataset in ['svhn', 'gtsrb']:
    #     args.eta = 1.0

    temp = []
    attack_rounds_txt = ''
    for attack_round in args.attack_rounds:
        temp.append(int(attack_round))
        attack_rounds_txt += '|' + str(attack_round)
    attack_rounds_txt += '|'
    args.attack_rounds = temp
    args.attack_rounds_txt = attack_rounds_txt

    args.ar_lmd = float(args.ar_lmd)
    args.alpha = float(args.alpha)
    # # set args.eugene_scaling
    # if args.p_dataset in ['cifar10', 'tinyimagenet100']:
    #     args.eugene_scaling = 10.0
    # elif args.p_dataset in ['svhn', 'gtsrb']:
    #     args.eugene_scaling = 5.0
    

class Rand_Augment():
    def __init__(self, Numbers=None, max_Magnitude=None):
        self.transforms = ['autocontrast', 'equalize', 'rotate', 'solarize', 'color', 'posterize',
                           'contrast', 'brightness', 'sharpness', 'shearX', 'shearY', 'translateX', 'translateY']
        if Numbers is None:
            self.Numbers = len(self.transforms) // 2
        else:
            self.Numbers = Numbers
        if max_Magnitude is None:
            self.max_Magnitude = 10
        else:
            self.max_Magnitude = max_Magnitude
        fillcolor = 128
        self.ranges = {
            # these  Magnitude   range , you  must test  it  yourself , see  what  will happen  after these  operation ,
            # it is no  need to obey  the value  in  autoaugment.py
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 0.2, 10),
            "translateY": np.linspace(0, 0.2, 10),
            "rotate": np.linspace(0, 360, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int_),
            "solarize": np.linspace(256, 231, 10),
            "contrast": np.linspace(0.0, 0.5, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.3, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,           
            "invert": [0] * 10
        }
        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fill=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fill=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fill=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fill=fillcolor),
            "rotate": lambda img, magnitude: self.rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: img,
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

    def rand_augment(self):
        """Generate a set of distortions.
             Args:
             N: Number of augmentation transformations to apply sequentially. N  is len(transforms)/2  will be best
             M: Max_Magnitude for all the transformations. should be  <= self.max_Magnitude """

        M = np.random.randint(0, self.max_Magnitude, self.Numbers)

        sampled_ops = np.random.choice(self.transforms, self.Numbers)
        return [(op, Magnitude) for (op, Magnitude) in zip(sampled_ops, M)]

    def __call__(self, image):
        operations = self.rand_augment()
        for (op_name, M) in operations:
            operation = self.func[op_name]
            mag = self.ranges[op_name][M]
            image = operation(image, mag)
        return image

    def rotate_with_fill(self, img, magnitude):
        #  I  don't know why  rotate  must change to RGBA , it is  copy  from Autoaugment - pytorch
        rot = img.convert("RGBA").rotate(magnitude)
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)


