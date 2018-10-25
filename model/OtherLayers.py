"""
This script contains some classes used in our models
"""

import numpy as np
import torch
import torch.nn as nn


class Identity(nn.Module):
    """
    An identity layer f(x) = x
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class LowerBoundClipper(object):
    def __init__(self, threshold):
        self.bound = threshold

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w[w < self.bound] = self.bound
        if hasattr(module, 'bias'):
            w = module.bias.data
            w[w < self.bound] = self.bound


class MaxLogLike(nn.Module):
    """
    The negative log-likelihood loss of events of point processes
    nll = sum_{i in batch}[ -log lambda_ci(ti) + sum_c Lambda_c(ti) ]
    """
    def __init__(self):
        super(MaxLogLike, self).__init__()
        self.eps = float(np.finfo(np.float32).eps)

    def forward(self, lambda_t, Lambda_t, c):
        """
        compute negative log-likelihood of the given batch
        :param lambda_t: (batchsize, 1) float tensor representing intensity functions
        :param Lambda_t: (batchsize, num_type) float tensor representing the integration of intensity in [t_i-1, t_i]
        :param c: (batchsize, 1) long tensor representing the types of events
        :return: nll (1,)  float tensor representing negative log-likelihood
        """
        return -(lambda_t+self.eps).log().sum() + Lambda_t.sum()


class MaxLogLikePerSample(nn.Module):
    """
    The negative log-likelihood loss of events of point processes
    nll = [ -log lambda_ci(ti) + sum_c Lambda_c(ti) ]
    """
    def __init__(self):
        super(MaxLogLikePerSample, self).__init__()
        self.eps = float(np.finfo(np.float32).eps)

    def forward(self, lambda_t, Lambda_t, c):
        """
        compute negative log-likelihood of the given batch
        :param lambda_t: (batchsize, 1) float tensor representing intensity functions
        :param Lambda_t: (batchsize, num_type) float tensor representing the integration of intensity in [t_i-1, t_i]
        :param c: (batchsize, 1) long tensor representing the types of events
        :return: nll (batchsize,)  float tensor representing negative log-likelihood
        """
        return -(lambda_t[:, 0]+self.eps).log() + Lambda_t.sum(1)


class LeastSquare(nn.Module):
    """
    The least-square loss of events of point processes
    ls = || Lambda_c(t) - N(t) ||_F^2
    """
    def __init__(self):
        super(LeastSquare, self).__init__()
        self.ls_loss = nn.MSELoss()

    def forward(self, lambda_t, Lambda_t, c):
        """
        compute least-square loss between integrated intensity and counting matrix
        :param lambda_t: (batch_size, 1)
        :param Lambda_t: (batch_size, num_type)
        :param c: (batch_size, 1)
        :return:
        """
        mat_onehot = torch.zeros(Lambda_t.size(0), Lambda_t.size(1)).scatter_(1, c, 1)
        # mat_onehot = mat_onehot.type(torch.FloatTensor)
        # print(Lambda_t.size())
        # print(mat_onehot.size())
        return self.ls_loss(Lambda_t, mat_onehot)


class CrossEntropy(nn.Module):
    """
    The cross entropy loss that maximize the conditional probability of current event given its intensity
    ls = -sum_{i in batch} log p(c_i | t_i, c_js, t_js)
    """
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.entropy_loss = nn.CrossEntropyLoss()

    def forward(self, lambda_t, Lambda_t, c):
        return self.entropy_loss(Lambda_t, c[:, 0])




