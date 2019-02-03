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
        # if hasattr(module, 'bias'):
        #     w = module.bias.data
        #     w[w < self.bound] = self.bound


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


class GromovWassersteinDiscrepancy(nn.Module):
    """
    Calculate Gromov-Wasserstein discrepancy given optimal transport and cost matrix
    """
    def __init__(self, loss_type):
        super(GromovWassersteinDiscrepancy, self).__init__()
        self.loss_type = loss_type

    def forward(self, As, At, Trans_st, p_s, p_t):
        """
        Calculate GW discrepancy
        :param As: learnable cost matrix of source
        :param At: learnable cost matrix of target
        :param Trans_st: the fixed optimal transport
        :param p_s: the fixed distribution of source
        :param p_t: the fixed distribution of target
        :return: dgw
        """
        ns = p_s.size(0)
        nt = p_t.size(0)
        if self.loss_type == 'L2':
            # f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = 2b
            # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
            # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
            f1_st = torch.matmul(As ** 2, p_s).repeat(1, nt)
            f2_st = torch.matmul(torch.t(p_t), torch.t(At ** 2)).repeat(ns, 1)
            cost_st = f1_st + f2_st
            cost = cost_st - 2 * torch.matmul(torch.matmul(As, Trans_st), torch.t(At))
        else:
            # f1(a) = a*log(a) - a, f2(b) = b, h1(a) = a, h2(b) = log(b)
            # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
            # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
            f1_st = torch.matmul(As * torch.log(As + 1e-5) - As, p_s).repeat(1, nt)
            f2_st = torch.matmul(torch.t(p_t), torch.t(At)).repeat(ns, 1)
            cost_st = f1_st + f2_st
            cost = cost_st - torch.matmul(torch.matmul(As, Trans_st), torch.t(torch.log(At + 1e-5)))

        d_gw = (cost * Trans_st).sum()
        return d_gw


class WassersteinDiscrepancy(nn.Module):
    """
    Calculate Wasserstein discrepancy given optimal transport and
    """
    def __init__(self, loss_type):
        super(WassersteinDiscrepancy, self).__init__()
        self.loss_type = loss_type

    def forward(self, mu_s, mu_t, Trans_st, p_s, p_t):
        """
        Calculate GW discrepancy
        :param mu_s: learnable base intensity of source
        :param mu_t: learnable base intensity of target
        :param Trans_st: the fixed optimal transport
        :param p_s: the fixed distribution of source
        :param p_t: the fixed distribution of target
        :return: dgw
        """
        ns = p_s.size(0)
        nt = p_t.size(0)
        if self.loss_type == 'L2':
            # f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = 2b
            # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
            # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
            f1_st = (mu_s ** 2).repeat(1, nt)
            f2_st = torch.t(mu_t ** 2).repeat(ns, 1)
            cost_st = f1_st + f2_st
            cost = cost_st - 2 * torch.matmul(mu_s, torch.t(mu_t))
        else:
            # f1(a) = a*log(a) - a, f2(b) = b, h1(a) = a, h2(b) = log(b)
            # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
            # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
            f1_st = (mu_s ** 2).repeat(1, nt)
            f2_st = torch.t(mu_t ** 2).repeat(ns, 1)
            cost = f1_st * torch.log(f1_st/(f2_st + 1e-5)) - f1_st + f2_st
        d_w = (cost * Trans_st).sum()
        return d_w





