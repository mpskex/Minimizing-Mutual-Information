import torch
from torch import nn
import torchvision
import numpy as np
from progress.bar import Bar
from copy import deepcopy

def set_device(device=-1):
    if type(device) is int:
        if device == -1:
            return "cuda" if torch.cuda.is_available() else "cpu"
        else:
            return "cuda:"+str(device) if torch.cuda.is_available() else "cpu"
    elif type(device) is str:
        return device
    else:
        raise TypeError

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(
            *list(self.vgg.classifier.children())[:6])
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.vgg.features(x)
        x = x.view(x.size(0), -1)
        x = self.vgg.classifier(x)
        return x

def extract_feature(data_loader, model, name, device:int=None):
    model.eval()
    device = set_device(device if device is not None else -1)
    with torch.no_grad():
        with Bar("Gathering Database Set Features", max=len(data_loader)) as bar:
            features = []
            labels = []
            indexes = []
            model.eval()
            for i, (index, im, l) in enumerate(data_loader):
                im = im.to(device)
                x = model(im)
                x, l, index = x.cpu().detach(), l.detach(), index.detach()
                features.append(deepcopy(x))
                labels.append(deepcopy(l))
                indexes.append(deepcopy(index))
                del x, l, index
                bar.next()
            features = torch.cat(features, dim=0)
            labels = torch.cat(labels, dim=0)
            indexes = torch.cat(indexes, dim=0)
            assert indexes.size(0) == features.size(0) == labels.size(0)
            d = {'features': features.numpy(),
                'labels': labels.numpy(),
                'indexes': indexes.numpy()}
            np.savez(name + '.npz', **d)
        print('Saved!')

def extract_feature_cifar(cifar_data_loader, model, name, class_num=10, device=None):
    model.eval()
    device = set_device(device if device is not None else -1)
    with torch.no_grad():
        with Bar("Gathering Database Set Features", max=len(cifar_data_loader)) as bar:
            features = []
            labels = []
            indexes = []
            model.eval()
            for i, (index, im, l) in enumerate(cifar_data_loader):
                im = im.to(device)
                x = model(im)
                x, l, index = x.cpu().detach(), l.detach(), index.detach()
                features.append(deepcopy(x))
                labels.append(deepcopy(l))
                indexes.append(deepcopy(index))
                del x, l, index
                bar.next()
            features = torch.cat(features, dim=0)
            labels = torch.cat(labels, dim=0)
            indexes = torch.cat(indexes, dim=0)
            assert indexes.size(0) == features.size(0) == labels.size(0)
            d = {'features': features.numpy(),
                'labels': labels.numpy(),
                'indexes': indexes.numpy()}
            np.savez(name + '.npz', **d)
        print('Saved!')