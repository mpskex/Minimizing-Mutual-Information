import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.datasets as dsets
from torchvision import transforms
import torchvision
import math
import numpy as np
import argparse
import sys
from progress.bar import Bar


torch.multiprocessing.set_sharing_strategy('file_system')

p = argparse.ArgumentParser("Training script")
p.add_argument('--encode_length', default=32,
               type=int, help="Hash Code Length")

params = p.parse_args(sys.argv[1:])

# Hyper Parameters
num_epochs = 60
batch_size = 256
# epoch_lr_decrease = 300
learning_rate = 0.0001
#   Original 32 bits is 48.71
#   Original 16 bits is 45.39
encode_length = params.encode_length


if encode_length <= 16:
    num_epochs = 300

device = "cuda" if torch.cuda.is_available() else "cpu"

train_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset
train_dataset = dsets.CIFAR10(root='data/',
                              train=True,
                              transform=train_transform,
                              download=True)

test_dataset = dsets.CIFAR10(root='data/',
                             train=False,
                             transform=test_transform)

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          num_workers=4)

# new layer


class hash(Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # input,  = ctx.saved_tensors
        # grad_output = grad_output.data

        return grad_output


def hash_layer(input):
    return hash.apply(input)


class CNN(nn.Module):
    def __init__(self, encode_length):
        super(CNN, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(
            *list(self.vgg.classifier.children())[:6])
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.fc_encode = nn.Linear(4096, encode_length, bias=False)
        for param in self.fc_encode.parameters():
            torch.nn.init.kaiming_normal_(param, mode='fan_out')

    def forward(self, x):
        x = self.vgg.features(x)
        x = x.view(x.size(0), -1)
        x = self.vgg.classifier(x)
        return x



cnn = CNN(encode_length=encode_length).to(device)


# Train the Model
cnn.eval()

#   MI Loss part
with torch.no_grad():
    with Bar("Gathering Train Set Features", max=len(train_loader)) as bar:
        features = []
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device).float()
            x = cnn(images)
            x = torch.cat([x, labels.view(-1, 1)], dim=-1)
            features.append(x)
            bar.next()
        features = torch.cat(features, dim=0)
        np.save('cifar_feat.train.npy', features.cpu().detach().numpy())
    print('Saved!')

with torch.no_grad():
    with Bar("Gathering Train Set Features", max=len(test_loader)) as bar:
        features = []
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device).float()
            x = cnn(images)
            x = torch.cat([x, labels.view(-1, 1)], dim=-1)
            features.append(x)
            bar.next()
        features = torch.cat(features, dim=0)
        np.save('cifar_feat.test.npy', features.cpu().detach().numpy())
    print('Saved!')