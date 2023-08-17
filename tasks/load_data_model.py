from re import I
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms
import torch
import torchvision.datasets as dataset
import glob
from torch.utils.data import Dataset, DataLoader
from models.my_models import resnet_18, resnet_34, resnet_152, resnet_101, vgg_13, vgg_16, mobilenet_v2, RegNetX_200MF, PreActResNet101, shufflenet_v2, wideresnet, ShuffleNetG3, vgg_11, vgg_19, SENet18, EfficientNetB0, PNASNetB
from models.models_new import densenet121, resnet152, vgg19_bn, googlenet, shufflenetv2, inceptionv4, resnet34, resnext50, seresnet50 # wideresnet
from tasks.task import Task
import os
from attack import itr_merge
from utils.place365_parser import PlaceDataset
from torch._utils import _accumulate
from torch.utils.data import Subset, DataLoader, ConcatDataset

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

def select_emnist(data_loader, name, classes, num_imgs):
    A = {'1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0,}
    for batch_idx, (x, y) in enumerate(data_loader):
        a = 0
        if y.item() in classes:
            if A[str(y.item())] <= num_imgs:
                os.makedirs(f'XXXXX/Projects/Oakland22/dataset/emnist/{name}/{y.item()}', exist_ok=True)
                torchvision.utils.save_image(x, f'XXXXX/Projects/Oakland22/dataset/emnist/{name}/{y.item()}/{A[str(y.item())]}.png')        
                A[str(y.item())] += 1
        for i in range(11):
            if i == 0:
                pass
            else:
                if A[str(i)]>=num_imgs:
                    a += 1
        if a == 10:
            break
        print(A, a)

class Data_Model_Task(Task):
    def build_dataset(self, task, mode=None):
        if self.params.transform_train:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                # self.normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                # self.normalize,
            ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            # self.normalize,
        ])
        if task == 'mnist':
            train_dataset = torchvision.datasets.MNIST(
                        root=self.params.data_path,
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                # self.normalize,
                            ]))
            test_dataset = torchvision.datasets.MNIST(
                        root=self.params.data_path,
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                # self.normalize,
                            ]))
            self.A_classes = 10 * [1]
        elif task == 'kmnist':
            train_dataset = torchvision.datasets.KMNIST(
                        root=self.params.data_path,
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                # self.normalize,
                            ]))
            test_dataset = torchvision.datasets.KMNIST(
                        root=self.params.data_path,
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                # self.normalize,
                            ]))
            self.A_classes = 10 * [1]
        elif task == 'qmnist':
            train_dataset = torchvision.datasets.QMNIST(
                        root=self.params.data_path,
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                # self.normalize,
                            ]))
            test_dataset = torchvision.datasets.QMNIST(
                        root=self.params.data_path,
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                # self.normalize,
                            ]))
            self.A_classes = 10 * [1]

        elif task == 'emnist_select':
            train_dataset_inner = torchvision.datasets.EMNIST(
                        root=self.params.data_path,
                        split='letters',
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                # self.normalize,
                            ]))
            test_dataset_inner = torchvision.datasets.EMNIST(
                        root=self.params.data_path,
                        split='letters',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                # self.normalize,
                            ]))
            self.A_classes = 10 * [1]

            train_loader_inner = DataLoader(train_dataset_inner,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=2)
            
            test_loader_inner = DataLoader(test_dataset_inner,
                                          batch_size=1,
                                          shuffle=False, num_workers=2)

            select_emnist(train_loader_inner, 'train', [1,2,3,4,5,6,7,8,9,10], num_imgs=5000)
            select_emnist(test_loader_inner, 'test', [1,2,3,4,5,6,7,8,9,10], num_imgs=1000)
            # print(len(train_dataset), len(test_dataset))
            exit()
        elif task == 'fashion_mnist':
            train_dataset = torchvision.datasets.FashionMNIST(
                        root=self.params.data_path,
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                # self.normalize,
                            ]))
            test_dataset = torchvision.datasets.FashionMNIST(
                        root=self.params.data_path,
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                # self.normalize,
                            ]))
            self.A_classes = 10 * [1]
        elif task == 'cifar10':
            if mode==None:
                train_dataset = torchvision.datasets.CIFAR10(
                            root=self.params.data_path,
                            train=True,
                            download=True,
                            transform=transform_train)
            elif mode == 'convert':
                train_dataset = torchvision.datasets.CIFAR10(
                            root=self.params.data_path,
                            train=True,
                            download=True,
                            transform=transform_test)
            test_dataset = torchvision.datasets.CIFAR10(
                        root=self.params.data_path,
                        train=False,
                        download=True,
                        transform=transform_test)
            self.A_classes = 10 * [1]
        elif task == 'cifar100':
            train_dataset = torchvision.datasets.CIFAR100(
                        root=self.params.data_path,
                        train=True,
                        download=True,
                        transform=transform_train)
            test_dataset = torchvision.datasets.CIFAR100(
                        root=self.params.data_path,
                        train=False,
                        download=True,
                        transform=transform_test)
            self.A_classes = 100 * [1]
        # elif task == 'food101':
        #     train_dataset = torchvision.datasets.Food101(
        #                 root=self.params.data_path,
        #                 split='train',
        #                 download=True,
        #                 transform=transform_train)
        #     test_dataset = torchvision.datasets.Food101(
        #                 root=self.params.data_path,
        #                 split='test',
        #                 download=True,
        #                 transform=transform_test)
        #     self.A_classes = 101 * [1]
        elif task == 'emnist':
            train_dir = 'XXXXX/Projects/Oakland22/dataset/emnist/train'
            valid_dir = 'XXXXX/Projects/Oakland22/dataset/emnist/test'
            train_dataset =  torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
            test_dataset =  torchvision.datasets.ImageFolder(valid_dir, transform=transform_test)
            self.A_classes = 10 * [1]
        elif task == 'tinyimagenet':
            train_dir = 'XXXXXdataset/common_data/tiny-imagenet-200/train'
            valid_dir = 'XXXXXdataset/common_data/tiny-imagenet-200/val/images'
            train_dataset =  torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
            test_dataset =  torchvision.datasets.ImageFolder(valid_dir, transform=transform_test)
            self.A_classes = 200 * [1]
        elif task == 'tinyimagenet100':
            train_dir = 'XXXXX/dataset/tiny-imagenet-100/train'
            valid_dir = 'XXXXX/dataset/tiny-imagenet-100/val'
            train_dataset =  torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
            test_dataset =  torchvision.datasets.ImageFolder(valid_dir, transform=transform_test)
            self.A_classes = 100 * [1]
        elif task == 'imagenet32':
            train_dataset = ImageNet32(root='XXXXX/Projects/Oakland22/dataset/imagenet32', train=True, transform=transform_train)
            test_dataset = ImageNet32(root='XXXXX/Projects/Oakland22/dataset/imagenet32', train=False, transform=transform_test)
            self.A_classes = 1000 * [1]
        elif task == 'celeba':
            # define custom dataloader from torch
            class celeba(Dataset):
                def __init__(self, data_path=None, label_path=None):
                    self.data_path = data_path
                    self.label_path = label_path
                def __len__(self):
                    return len(self.data_path)
                
                def __getitem__(self, idx):
                    image_set = Image.open(self.data_path[idx])
                    image_tensor = transform_test(image_set)
                    image_label = torch.Tensor(self.label_path[idx])

                    return image_tensor, image_label

            data_path = sorted(glob.glob('XXXXX/Projects/Oakland22/dataset/img_align_celeba/*.jpg'))
            # print(len(data_path))

            # get the label of images
            label_path = "XXXXXdataset/CelebA/Anno/list_attr_celeba.txt"
            label_list = open(label_path).readlines()[2:]
            data_label = []
            for i in range(len(label_list)):
                data_label.append(label_list[i].split())

            # transform label into 0 and 1
            for m in range(len(data_label)):
                data_label[m] = [n.replace('-1', '0') for n in data_label[m]][1:]
                data_label[m] = [int(p) for p in data_label[m]]

            # get the attributes names for display
            attributes = open(label_path).readlines()[1].split()

            dataset = celeba(data_path, data_label)
            # split data into train, valid, test set 7:2:1
            indices = list(range(202599))
            split_train = 50000
            test_valid = 60000
            train_idx, test_idx, _ = indices[:split_train], indices[split_train:test_valid], indices[test_valid:]

            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(test_idx)

            trainloader = torch.utils.data.DataLoader(dataset, batch_size=200, sampler=train_sampler)
            testloader =  torch.utils.data.DataLoader(dataset, batch_size=200, sampler=test_sampler)

            batch_id = 0
            for x, y in itr_merge([trainloader]):
                input = x if batch_id == 0 else torch.cat((input, x), 0)
                label = y if batch_id == 0 else torch.cat((label, y), 0)
                batch_id = batch_id + 1
            train_dataset = torch.utils.data.TensorDataset(input, label.float())
            torch.save(train_dataset, 'XXXXX/Projects/Oakland22/dataset/celeba/train_dataset.pt')

            batch_id = 0
            for x, y in itr_merge([testloader]):
                input = x if batch_id == 0 else torch.cat((input, x), 0)
                label = y if batch_id == 0 else torch.cat((label, y), 0)
                batch_id = batch_id + 1
            test_dataset = torch.utils.data.TensorDataset(input, label.float())
            torch.save(test_dataset, 'XXXXX/Projects/Oakland22/dataset/celeba/test_dataset.pt')
            self.A_classes = 40 * [1]

        elif task == 'celeba':
            train_dataset = torch.load('XXXXX/Projects/Oakland22/dataset/celeba/train_dataset.pt')
            test_dataset = torch.load('XXXXX/Projects/Oakland22/dataset/celeba/test_dataset.pt')
            self.A_classes = 40 * [1]

        elif task == 'stl10':
            train_dataset = torchvision.datasets.STL10(
                        root=self.params.data_path,
                        split='train',
                        download=True,
                        transform=transform_train)
            test_dataset = torchvision.datasets.STL10(
                        root=self.params.data_path,
                        split='test', 
                        download=True, 
                        transform=transform_test)
            self.A_classes = 10 * [1]
        elif task == 'ag_news':
            import torchtext
            train_dataset = torchtext.datasets.AG_NEWS(
                        root=self.params.data_path,
                        split='train',
                        download=True,
                        )
            test_dataset = torchtext.datasets.AG_NEWS(
                        root=self.params.data_path,
                        split='test', 
                        download=True, 
                        )
            self.A_classes = 4 * [1]
            exit()
        elif 'place' in task:
            if task == "place100":
                dataset = PlaceDataset(
                    root='XXXXX/dataset', transform=transform_train, dataset_name="place100")
                self.A_classes = 100 * [1]
            elif task == "place80":
                dataset = PlaceDataset(
                    root='XXXXX/dataset', transform=transform_train, dataset_name="place80")
                self.A_classes = 80 * [1]
            elif task == "place60":
                dataset = PlaceDataset(
                    root='XXXXX/dataset', transform=transform_train, dataset_name="place60")
                self.A_classes = 60 * [1]
            elif task == "place40":
                dataset = PlaceDataset(
                    root='XXXXX/dataset', transform=transform_train, dataset_name="place40")
                self.A_classes = 40 * [1]
            elif task == "place20":
                dataset = PlaceDataset(
                    root='XXXXX/dataset', transform=transform_train, dataset_name="place20")
                self.A_classes = 20 * [1]
            fulll_length = len(dataset)
            train_size = 55000
            test_size = fulll_length - train_size

            train_dataset, test_dataset = dataset_split(dataset, [train_size, test_size])
        return train_dataset, test_dataset

    def build_model(self) -> nn.Module:
        # if self.params.pretrained:
        #     model = resnet_18(pretrained=True)

        #     # model is pretrained on ImageNet changing classes to CIFAR
        #     model.fc = nn.Linear(512, len(self.A_classes))
        # else:
        # if self.params.model_arch == 'resnet_34':
        #     model = resnet_34(num_classes=len(self.A_classes))
        # elif self.params.model_arch == 'vgg_16':
        #     model = vgg_16(num_classes=len(self.A_classes))
        # elif self.params.model_arch == 'vgg_19':
        #     model = vgg_19(num_classes=len(self.A_classes))
        # elif self.params.model_arch == 'mobilenet_v2':
        #     model = mobilenet_v2(num_classes=len(self.A_classes))
        # elif self.params.model_arch == 'wideresnet':
        #     model = wideresnet(num_classes=len(self.A_classes))
        # elif self.params.model_arch == 'resnet_152':
        #     model = resnet_152(num_classes=len(self.A_classes))
        # elif self.params.model_arch == 'resnet_101':  resnet34
        #     model = resnet_101(num_classes=len(self.A_classes))
        if self.params.model_arch == 'resnet34':
            model = resnet34(num_classes=len(self.A_classes)) 
        # elif self.params.model_arch == 'resnet_101':
        #     model = resnet_101(num_classes=len(self.A_classes)) 
        elif self.params.model_arch == 'densenet121':
            model = densenet121(num_classes=len(self.A_classes))
        elif self.params.model_arch == 'inceptionv4':
            model = inceptionv4(num_classes=len(self.A_classes))
        elif self.params.model_arch == 'shufflenetv2':
            model = shufflenetv2(num_classes=len(self.A_classes))
        elif self.params.model_arch == 'wideresnet':
            model = wideresnet(num_classes=len(self.A_classes))
        elif self.params.model_arch == 'mobilenet_v2':
            model = mobilenet_v2(num_classes=len(self.A_classes))
        elif self.params.model_arch == 'resnext50':
            model = resnext50(num_classes=len(self.A_classes))
        elif self.params.model_arch == 'regnetx_200mf':
            model = RegNetX_200MF(num_classes=len(self.A_classes))
        elif self.params.model_arch == 'preactnet101':
            model = PreActResNet101(num_classes=len(self.A_classes))
        return model

    def remove_semantic_backdoors(self):
        """
        Semantic backdoors still occur with unmodified labels in the training
        set. This method removes them, so the only occurrence of the semantic
        backdoor will be in the
        :return: None
        """

        all_images = set(range(len(self.train_dataset)))
        unpoisoned_images = list(all_images.difference(set(
            self.params.poison_images)))

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       sampler=SubsetRandomSampler(
                                           unpoisoned_images))


import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

# from .utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset


class ImageNet32(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    # base_folder = "cifar-10-batches-py"
    # url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    # filename = "cifar-10-python.tar.gz"
    # tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["train_data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["train_data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["train_data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["train_data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["train_data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
        # ["train_data_batch_6", "482c414d41f54cd18b22e5b47cb7c3cb"],
        # ["train_data_batch_7", "482c414d41f54cd18b22e5b47cb7c3cb"],
        # ["train_data_batch_8", "482c414d41f54cd18b22e5b47cb7c3cb"],
        # ["train_data_batch_9", "482c414d41f54cd18b22e5b47cb7c3cb"],
        # ["train_data_batch_10", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["val_data", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
       
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        # if download:
        #     self.download()

        # if not self._check_integrity():
        #     raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        # self._load_meta()

    # def _load_meta(self) -> None:
    #     path = os.path.join(self.root, self.base_folder, self.meta["filename"])
    #     if not check_integrity(path, self.meta["md5"]):
    #         raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
    #     with open(path, "rb") as infile:
    #         data = pickle.load(infile, encoding="latin1")
    #         self.classes = data[self.meta["key"]]
    #     self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target-1


    def __len__(self) -> int:
        return len(self.data)

    # def _check_integrity(self) -> bool:
    #     for filename, md5 in self.train_list + self.test_list:
    #         fpath = os.path.join(self.root, self.base_folder, filename)
    #         if not check_integrity(fpath, md5):
    #             return False
    #     return True

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

