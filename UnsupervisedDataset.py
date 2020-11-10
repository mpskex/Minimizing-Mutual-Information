import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets.folder import default_loader


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class UnsupervisedDataset(torch.utils.data.Dataset):

    def __init__(self, root,
                 transform=None, 
                 target_transform=None, 
                 train=True, 
                 database_bool=False, 
                 with_label=False,):
        self.loader = default_loader
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.base_folder = 'train.txt'
        elif database_bool:
            self.base_folder = 'database.txt'
        else:
            self.base_folder = 'test.txt'

        self.train_data = []
        self.train_labels = []

        filename = os.path.join(self.root, self.base_folder)

        num_classes = None
        with open(filename, 'r') as file_to_read:
            for lines in file_to_read.readlines():
                splitted = lines.split(' ')
                pos_tmp = os.path.join(self.root, splitted[0])
                label_tmp = splitted[1:]
                if num_classes is None:
                    num_classes = len(label_tmp)
                else:
                    assert num_classes == len(label_tmp)
                self.train_data.append(pos_tmp)
                self.train_labels.append(label_tmp)
        self.train_labels = np.array(self.train_labels, dtype=np.float)
        self.num_classes = num_classes
        print(self.train_labels.shape)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.train_data[index], self.train_labels[index]

        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        return index, img, target

    def __len__(self):
        return len(self.train_data)