import os
import torch
import torch.nn as nn
import torchvision
from UnsupervisedDataset import UnsupervisedDataset, test_transform
from collect import extract_feature, CNN, set_device
import argparse

p = argparse.ArgumentParser("Collect feature")
p.add_argument("--gpu", type=int, default=0, help="Designated GPU ID")
params = p.parse_args()
device = set_device(params.gpu)

# Hyper Parameters
batch_size = 256

# name = 'nuswide_81'
name = 'nuswide_21_m'

# Dataset
train_dataset = UnsupervisedDataset(root=os.path.join('data', name),
                                    train=True,
                                    transform=test_transform)

test_dataset = UnsupervisedDataset(root=os.path.join('data', name),
                                   train=False,
                                   transform=test_transform)

database_dataset = UnsupervisedDataset(root=os.path.join('data', name),
                                       train=False,
                                       transform=test_transform,
                                       database_bool=True)

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers=16)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          num_workers=16)

database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                              batch_size=batch_size,
                                              num_workers=16)

cnn = CNN().to(device)

# Train the Model

extract_feature(train_loader, cnn, 'nuswide_feat.train', device=device)
extract_feature(database_loader, cnn, 'nuswide_feat.db', device=device)
extract_feature(test_loader, cnn, 'nuswide_feat.test', device=device)
