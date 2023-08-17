import json
import torchvision.transforms as transforms
import torchvision
import PIL.Image as Image
import os
import torch
import torch.nn as nn
torch.manual_seed(0)
# from utils.dataset_parser.dataset_processing import prepare_dataset, count_dataset


class PlaceDataset(torch.utils.data.Dataset):
    # 80 * 750 = 60,000
    def __init__(self, root, transform=None, dataset_name="place80"):
        self.root = root
        self.transform = transform
        self.base_path = os.path.join(root, dataset_name)
        self.filename_path = os.path.join(
            self.base_path, "image_index_label_list.json")
        self.filename_list = json.loads(open(self.filename_path).read())
        # [["place100/data/0_30_1.jpg", 30, 1], ["place100/data/1_30_1.jpg", 30, 1], ["place100/data/2_30_1.jpg", 30, 1], [image_path, category, indoor/outdoor]
        self.category_label_index_dict = {}
        self.dataset_label_index_dict = {
                                        "place20":[160, 65, 69, 134, 6, 232, 170, 302, 143, 179, 55, 87, 215, 21, 89, 22, 182, 311, 158, 95],
                                        "place40":[134, 6, 138, 141, 143, 145, 147, 21, 22, 277, 158, 160, 170, 299, 302, 179, 308, 182, 55, 311, 313, 186, 60, 65, 69, 70, 197, 327, 208, 83, 342, 87, 215, 89, 218, 92, 95, 232, 235, 252],
                                        "place60":[133, 134, 6, 137, 138, 141, 143, 144, 145, 18, 147, 273, 21, 22, 277, 156, 157, 158, 31, 160, 288, 170, 299, 302, 179, 308, 182, 55, 311, 313, 186, 183, 60, 65, 196, 69, 70, 197, 327, 324, 199, 331, 208, 83, 342, 87, 215, 89, 218, 92, 95, 224, 353, 231, 232, 360, 235, 117, 252, 255],
                                        "place80":[0, 4, 133, 134, 6, 135, 137, 138, 141, 270, 143, 144, 145, 18, 147, 273, 21, 22, 277, 278, 156, 157, 158, 31, 160, 288, 161, 284, 163, 170, 299, 302, 303, 179, 308, 307, 182, 55, 311, 313, 186, 183, 60, 59, 63, 65, 196, 69, 70, 197, 327, 324, 199, 331, 202, 208, 83, 84, 341, 342, 87, 215, 89, 218, 219, 92, 344, 343, 95, 224, 353, 349, 356, 231, 232, 360, 235, 117, 252, 255],
                                        "place100":[160, 134, 158, 215, 232, 95, 65, 143, 87, 311, 302, 179, 55, 21, 170, 89, 22, 6, 182, 69, 308, 70, 145, 277, 299, 186, 92, 342, 197, 218, 138, 60, 327, 147, 83, 313, 235, 141, 208, 252, 137, 231, 18, 156, 117, 288, 273, 196, 144, 255, 353, 31, 157, 183, 224, 331, 133, 324, 360, 199, 219, 84, 344, 0, 343, 349, 278, 59, 303, 63, 161, 284, 341, 4, 270, 307, 356, 202, 135, 163, 155, 298, 28, 85, 209, 223, 96, 201, 75, 354, 103, 265, 330, 210, 90, 67, 38, 336, 132, 61],
                                        }
        self.category_label_index_dict = {self.dataset_label_index_dict[dataset_name][i]:i for i in range(len(self.dataset_label_index_dict[dataset_name]))}
        # selected_label = set([row[1] for row in self.filename_list])
        # # print([row[1] for row in self.filename_list])
        # import numpy as np
        # import random
        # np.random.seed(1)
        # random.seed(1)
        # print(selected_label)
        # exit()
        # index = 0
        # for original_label in selected_label:
        #     self.category_label_index_dict[original_label] = index
        #     index += 1

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):

        # image_path = os.path.join(self.base_path, self.filename_list[index][0][2:])
        image_path = os.path.join(self.root, self.filename_list[index][0])
        image = Image.open(image_path).convert('RGB')

        target = {"category": self.category_label_index_dict[self.filename_list[index][1]],
                  "io": self.filename_list[index][2]}
        target = self.category_label_index_dict[self.filename_list[index][1]]

        if self.transform:
            image = self.transform(image)

        return image, target


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    root = "/workspace/projects/robust_mia/data/"
    dataset = PlaceDataset(
        root=root, transform=transform, dataset_name="place20")
    print(dataset[0])
    print(dataset[100])
    print(dataset[1000])
    print(len(dataset))
    print(dataset.category_label_index_dict)

    # target_train, target_test, shadow_train, shadow_test = prepare_dataset(dataset)
    # target_train_loader = torch.utils.data.DataLoader(target_train, batch_size=32, shuffle=True, num_workers=2)
    # target_test_loader = torch.utils.data.DataLoader(target_test, batch_size=32, shuffle=True, num_workers=2)
    # shadow_train_loader = torch.utils.data.DataLoader(shadow_train, batch_size=32, shuffle=True, num_workers=2)
    # shadow_test_laoder = torch.utils.data.DataLoader(shadow_test, batch_size=32, shuffle=True, num_workers=2)
    # all_data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    # count_dataset(target_train_loader, target_test_loader, shadow_train_loader, shadow_test_laoder, num_classes=100, attr="category")

