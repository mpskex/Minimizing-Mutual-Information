import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.autograd import Variable, grad
from torch.distributions import uniform
"""
Quantization Layers

Fangrui Liu @ University of British Columbia
fangrui.liu@ubc.ca

Copyright reserved 2020
"""
class StraightThroughEstimator(Function):
    @staticmethod
    def forward(ctx, input):
        result = input.sign()
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BinaryQuantization(nn.Module):
    def __init__(self, cfg, momentum=0.9, epsilon=1e-05):
        super().__init__()
        self.momentum = momentum
        self.cfg = cfg
        self.epsilon = epsilon
        self.mode = 0
        if not self.cfg.with_fc:
            self.out_dim = self.cfg.zdim // 2 ** (self.cfg.layers + 1)
        else:
            self.out_dim = self.cfg.zdim

        self.register_buffer('_lambda', torch.zeros(self.out_dim))
        self.register_buffer('_gamma', torch.ones(self.out_dim))


    def forward(self, input):
        if self.training:
            if not self.cfg.with_fc:
                mean = input.mean([0, 2, 3])
                var = input.var([0, 2, 3])
            else:
                mean = input.mean([0])
                var = input.var([0])
            with torch.no_grad():
                self._lambda.copy_((self.momentum * self._lambda) + (1.0-self.momentum) * mean)
                self._gamma.copy_((self.momentum * self._gamma) + (1.0-self.momentum) *
                                  (input.shape[0]/(input.shape[0]-1)*var))
        else:
            mean = self._lambda
            var = self._gamma

        # change shape
        if not self.cfg.with_fc:
            current_mean = mean.view([1, self.out_dim, 1, 1]).expand_as(input)
            current_var = var.view([1, self.out_dim, 1, 1]).expand_as(input)
        else:
            current_mean = mean.view([1, self.out_dim]).expand_as(input)
            current_var = var.view([1, self.out_dim]).expand_as(input)

        # get output
        y = (input - current_mean) / (current_var + self.epsilon).sqrt()
        y = StraightThroughEstimator.apply(y)
        return y

    def inverse(self, y):
        return y * self._gamma + self._lambda


class BinaryDequantization(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mode = 0
        if not self.cfg.with_fc:
            self.out_dim = self.cfg.zdim // 2 ** (self.cfg.layers + 1)
        else:
            self.out_dim = self.cfg.zdim
        # initialize weight(gamma), bias(beta), running mean and variance
        U = uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        self.gamma_star = nn.Parameter(U.sample(torch.Size([self.out_dim])).view(self.out_dim))
        self.lambda_star = nn.Parameter(torch.zeros(self.out_dim))

    def forward(self, input):
        if not self.cfg.with_fc:
            current_gamma_star = self.gamma_star.view([1, self.out_dim, 1, 1]).expand_as(input)
            current_lambda_star = self.lambda_star.view([1, self.out_dim, 1, 1]).expand_as(input)
        else:
            current_gamma_star = self.gamma_star.view([1, self.out_dim]).expand_as(input)
            current_lambda_star = self.lambda_star.view([1, self.out_dim]).expand_as(input)

        # get output
        y = current_gamma_star * input + current_lambda_star
        return y
