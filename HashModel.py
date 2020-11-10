import torch
from torch import nn
from torch.autograd import Function


from Quantization import BinaryQuantization

class HashModel(nn.Module):
    def __init__(self, encode_length, bin_layer, feat_dim=4096):
        super().__init__()
        self.fc_encode = nn.Linear(feat_dim, encode_length)
        self.bin_layer = bin_layer

    def forward(self, x):
        h = self.fc_encode(x)
        h, b = self.bin_layer(h)
        return x, h, b

def get_norm_layer(encode_length, xi=1, momentum=0.999):
    return BinaryQuantization(encode_length, xi=xi, momentum=momentum)


def get_standard_layer():
    class hash(Function):
        @staticmethod
        def forward(ctx, input):
            # ctx.save_for_backward(input)
            return torch.sign(input)
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output
    class hash_layer(nn.Module):
        def forward(self, x):
            return x, hash.apply(x)
    return hash_layer()

def get_hashmodel(encode_length, norm=True, feat_dim=4096, xi=1, momentum=0.999):
    if norm:
        bin_layer = get_norm_layer(encode_length, xi=xi, momentum=momentum)
    else:
        bin_layer = get_standard_layer()
    return HashModel(encode_length, bin_layer, feat_dim)