import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import argparse
import sys

from cal_map import calculate_top_map, compress, mean_average_precision
from PrecompDataset import PrecomputedDataset
from MutualInformation import MutualInformation

from torch.optim.lr_scheduler import MultiStepLR

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

p = argparse.ArgumentParser("Training script")
p.add_argument('--encode_length', default=16,
               type=int, help="Hash Code Length")
p.add_argument('--mi_loss', action="store_true",
               help="With mutual information loss or not")
p.add_argument('--mi_loss_weight', default=1e-4, type=float,
               help="With mutual information loss or not")
p.add_argument('--batch_size', default=32, type=int,
               help="Size of mini-batches")
p.add_argument('--ent_sort', action="store_true",
               help="If use a entropy sorted mutual information loss")
p.add_argument('--dataset', default="cifar10", type=str,
               help="If use a entropy sorted mutual information loss")
p.add_argument('--lr', default=1e-3, type=float,
               help="If use a entropy sorted mutual information loss")
p.add_argument('--gpu', default=0, type=int,
               help="If use a entropy sorted mutual information loss")

params = p.parse_args(sys.argv[1:])
print(params)
assert params.dataset in ['cifar10', 'nuswide', 'mscoco']

# Hyper Parameters
num_epochs = 300
batch_size = params.batch_size
# epoch_lr_decrease = 300
learning_rate = params.lr
#   Original 32 bits is 48.71
#   Original 16 bits is 45.39
#   Precomp 16 bits is 47.83
#   Precomp  8 bits with 1e-4@mi is 34.64
#   Precomp 16 bits with 1e-4@mi is 50.64
#   Precomp 32 bits with 1e-4@mi is 55.53
encode_length = params.encode_length
with_mi_loss = params.mi_loss


device = "cuda:"+str(params.gpu) if torch.cuda.is_available() else "cpu"


# Dataset
if params.dataset == 'cifar10':
    multi_label = False
    topk_map = 1000
    feat_dim = 4096
    train_dataset = PrecomputedDataset(
        'cifar_feat.train.npy', multi_label=multi_label)
    database_dataset = PrecomputedDataset(
        'cifar_feat.train.npy', multi_label=multi_label)
    test_dataset = PrecomputedDataset(
        'cifar_feat.test.npy', multi_label=multi_label)
elif params.dataset == 'nuswide':
    multi_label = True
    topk_map = 5000
    feat_dim = 4096
    train_dataset = PrecomputedDataset('nuswide_feat.train.npz')
    database_dataset = PrecomputedDataset('nuswide_feat.db.npz')
    test_dataset = PrecomputedDataset('nuswide_feat.test.npz')
elif params.dataset == 'mscoco':
    multi_label = True
    topk_map = 5000
    feat_dim = 4096
    train_dataset = PrecomputedDataset('mscoco_feat.train.npz')
    database_dataset = PrecomputedDataset('mscoco_feat.db.npz')
    test_dataset = PrecomputedDataset('mscoco_feat.test.npz')
elif params.dataset == 'mnist':
    multi_label = False
    topk_map = 1000
    feat_dim = 256
    train_dataset = PrecomputedDataset('mnist_feat.train.npz')
    database_dataset = PrecomputedDataset('mnist_feat.train.npz')
    test_dataset = PrecomputedDataset('mnist_feat.test.npz')


# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4)

database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
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
    def __init__(self, encode_length, feat_dim=4096):
        super(CNN, self).__init__()
        self.fc_encode = nn.Linear(feat_dim, encode_length)

    def forward(self, x):
        h = self.fc_encode(x)
        b = hash_layer(h)
        return x, h, b


mi_loss = MutualInformation(entropy_sorting=params.ent_sort).to(device)
cnn = CNN(encode_length=encode_length, feat_dim=feat_dim).to(device)


optimizer = torch.optim.SGD(cnn.fc_encode.parameters(),
                            lr=learning_rate, momentum=0.9, weight_decay=5e-4)
lr_scheduler = MultiStepLR(optimizer, [100, 250])
mi_optimizer = torch.optim.SGD(
    cnn.fc_encode.parameters(), lr=params.mi_loss_weight*learning_rate)
mi_lr_scheduler = MultiStepLR(mi_optimizer, [100, 250])

best = 0.0

# Train the Model
for epoch in range(num_epochs):
    cnn.train()
    #   MI Loss part
    if with_mi_loss:
        bits = []
        mi_optimizer.zero_grad()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            x, h, b = cnn(images)
            bits.append(b)
        bits = torch.cat(bits, dim=0)
        loss_mi = mi_loss(bits)
        loss_mi.backward()
        mi_optimizer.step()
        print('Epoch [%d/%d] MI Loss: %.12f' %
              (epoch+1, num_epochs, loss_mi.item()))
        mi_lr_scheduler.step()

    loss_aver = [0, 0]
    # adjust_learning_rate(optimizer, epoch)
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        x, h, b = cnn(images)

        target_b = F.cosine_similarity(
            b[:int(labels.size(0) / 2)], b[int(labels.size(0) / 2):])
        target_x = F.cosine_similarity(
            x[:int(labels.size(0) / 2)], x[int(labels.size(0) / 2):])
        loss1 = F.mse_loss(target_b, target_x)
        loss2 = torch.mean(
            torch.abs(torch.pow(torch.abs(h) - torch.ones(h.size()).to(device), 3)))
        loss = loss1 + 0.1 * loss2
        loss_aver[0] += loss1.item()
        loss_aver[1] += loss2.item()
        loss.backward()
        optimizer.step()
    lr_scheduler.step()
    _str = 'Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f Loss2: %.4f' % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size,
                                                                    loss_aver[0]/len(train_loader), loss_aver[1]/len(train_loader))
    print(_str)

    # Test the Model
    if (epoch + 1) % 5 == 0:
        cnn.eval()

        retrievalB, retrievalL, queryB, queryL = compress(
            database_loader, test_loader, cnn, multi_label=multi_label, device=device)
        print('---calculate top map---')
        result = mean_average_precision(
            queryB, retrievalB, queryL, retrievalL, R=topk_map)
        print(result)

        if result > best:
            best = result
            torch.save(cnn.state_dict(), 'exp/precomp_{}_b{}_beta{}.{}.pkl'.format("mi" if with_mi_loss else "cmp",
                                                                                   encode_length,
                                                                                   params.mi_loss_weight,
                                                                                   params.dataset))

        print('best: %.6f' % (best))

with torch.no_grad():
    bits = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        x, h, b = cnn(images)
        bits.append(b)
    bits = torch.cat(bits, dim=0)
    loss_mi = mi_loss(bits)
    print('Epoch [%d/%d] MI Loss: %.12f' %
          (epoch + 1, num_epochs, loss_mi.item()))
