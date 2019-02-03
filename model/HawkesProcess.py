"""
Hawkes process model and its variance

For the
The intensity function is formulated as
lambda_k(t) = mu_k + sum_{t_i<t} phi_{k,k_i}(t-t_i)

mu_k(t) has two options:
1) constant: mu_k, which can be implemented by an embedding layer
2) parametric model (neural network): mu_k = F(f_k), where f_k is the static feature of k-th event type

phi_{kk'}(t) has several options:
1) phi_{kk'}(t) = sum_i a_{kk'i} * kernel_i(t), where a_{kk'i} is coefficient and kernel_i(t) is predefined decay kernel
2) phi_{kk'}(t) = sum_i (u_{k}^T v_{k'}) * kernel_i(t), where u_k and v_{k'} are embedding vectors of event types
3) phi_{kk'}(t) = sum_i (w_{k}^T f_{k'}) * kernel_i(t), where w_k in R^D is the coefficient of the k-th event type
and f_{k'} is the feature of historical k'-th event type.
4) phi_{kk'}(t) = sum_i (f_{k}^T W f_{k'}) * kernel_i(t), where W in R^D*D is the bilinear alignment matrix
5) phi_{kk'}(t) = sum_i (F(f_k)^T F(f_{k'})) * kernel_i(t), where F(.) is a parametric model (neural network)
"""

import copy
import torch
import torch.nn as nn
from dev.util import logger
from model.OtherLayers import Identity, LowerBoundClipper, GromovWassersteinDiscrepancy, WassersteinDiscrepancy
from model.PointProcess import PointProcessModel
import model.ExogenousIntensityFamily
import model.EndogenousImpactFamily
import model.DecayKernelFamily
import numpy as np
from preprocess.DataOperation import samples2dict
import time


class HawkesProcessIntensity(nn.Module):
    """
    The class of inhomogeneous Poisson process
    """
    def __init__(self,
                 exogenous_intensity,
                 endogenous_intensity,
                 activation: str = None):
        super(HawkesProcessIntensity, self).__init__()
        self.exogenous_intensity = exogenous_intensity
        self.endogenous_intensity = endogenous_intensity
        if activation is None:
            self.intensity_type = "exogenous intensity + endogenous impacts"
            self.activation = 'identity'
        else:
            self.intensity_type = "{}(exogenous intensity + endogenous impacts)".format(activation)
            self.activation = activation

        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'softplus':
            self.act = nn.Softplus(beta=self.num_type**0.5)
        elif self.activation == 'identity':
            self.act = Identity()
        else:
            logger.warning('The actvation layer is {}, which can not be identified... '.format(self.activation))
            logger.warning('Identity activation is applied instead.')
            self.act = Identity()

    def print_info(self):
        logger.info('A generalized Hawkes process intensity:')
        logger.info('Intensity function lambda(t) = {}'.format(self.intensity_type))
        self.exogenous_intensity.print_info()
        self.endogenous_intensity.print_info()

    def forward(self, sample_dict):
        mu, Mu = self.exogenous_intensity(sample_dict)
        alpha, Alpha = self.endogenous_intensity(sample_dict)
        lambda_t = self.act(mu + alpha)  # (batch_size, 1)
        Lambda_T = self.act(Mu + Alpha)  # (batch_size, num_type)
        return lambda_t, Lambda_T

    def intensity(self, sample_dict):
        mu = self.exogenous_intensity.intensity(sample_dict)
        alpha = self.endogenous_intensity.intensity(sample_dict)
        lambda_t = self.act(mu + alpha)  # (batch_size, 1)
        # print('mu={}'.format(float(mu.sum())))
        # print('alpha={}'.format(float(alpha.sum())))
        return lambda_t

    def expect_counts(self, sample_dict):
        Mu = self.exogenous_intensity.expect_counts(sample_dict)
        Alpha = self.endogenous_intensity.expect_counts(sample_dict)
        Lambda_T = self.act(Mu + Alpha)
        return Lambda_T


class HawkesProcessModel(PointProcessModel):
    """
    The class of generalized Hawkes process model
    contains most of necessary function.
    """

    def __init__(self, num_type, mu_dict, alpha_dict, kernel_dict, activation, loss_type, use_cuda):
        """
        Initialize generalized Hawkes process
        :param num_type: int, the number of event types.
        :param mu_dict: the dictionary of exogenous intensity's setting
            mu_dict = {'model_name': the name of specific subclass of exogenous intensity,
                       'parameter_set': a dictionary contains necessary parameters}
        :param alpha_dict: the dictionary of endogenous intensity's setting
            alpha_dict = {'model_name': the name of specific subclass of endogenous impact,
                          'parameter_set': a dictionary contains necessary parameters}
        :param kernel_dict: the dictionary of decay kernel's setting
            kernel_dict = {'model_name': the name of specific subclass of decay kernel,
                           'parameter_set': a ndarray contains necessary parameters}
        :param activation: str, the type of activation function
        :param loss_type: str, the type of loss functions
            The length of the list is the number of modalities of the model
            Each element of the list is the number of event categories for each modality
        """
        super(HawkesProcessModel, self).__init__(num_type, mu_dict, loss_type, use_cuda)
        self.model_name = 'A Hawkes Process'
        # self.num_type = num_type
        self.activation = activation
        exogenousIntensity = getattr(model.ExogenousIntensityFamily, mu_dict['model_name'])
        endogenousImpacts = getattr(model.EndogenousImpactFamily, alpha_dict['model_name'])
        decayKernel = getattr(model.DecayKernelFamily, kernel_dict['model_name'])

        mu_model = exogenousIntensity(num_type, mu_dict['parameter_set'])
        kernel_para = kernel_dict['parameter_set'].to(self.device)
        kernel_model = decayKernel(kernel_para)
        alpha_model = endogenousImpacts(num_type, kernel_model, alpha_dict['parameter_set'])

        self.lambda_model = HawkesProcessIntensity(mu_model, alpha_model, self.activation)
        self.print_info()

    def plot_exogenous(self, sample_dict, output_name: str = None):
        intensity = self.lambda_model.exogenous_intensity.intensity(sample_dict)
        self.lambda_model.exogenous_intensity.plot_and_save(intensity, output_name)

    def plot_causality(self, sample_dict, output_name: str = None):
        infectivity = self.lambda_model.endogenous_intensity.granger_causality(sample_dict)
        self.lambda_model.endogenous_intensity.plot_and_save(infectivity, output_name)


class HawkesProcessModel_OT(PointProcessModel):
    """
    The class of generalized Hawkes process model
    contains most of necessary function.
    """
    def __init__(self, num_type, mu_dict, alpha_dict, kernel_dict, activation, loss_type, cost_type, use_cuda):
        """
        Initialize generalized Hawkes process
        :param num_type: int, the number of event types.
        :param mu_dict: the dictionary of exogenous intensity's setting
            mu_dict = {'model_name': the name of specific subclass of exogenous intensity,
                       'parameter_set': a dictionary contains necessary parameters}
        :param alpha_dict: the dictionary of endogenous intensity's setting
            alpha_dict = {'model_name': the name of specific subclass of endogenous impact,
                          'parameter_set': a dictionary contains necessary parameters}
        :param kernel_dict: the dictionary of decay kernel's setting
            kernel_dict = {'model_name': the name of specific subclass of decay kernel,
                           'parameter_set': a ndarray contains necessary parameters}
        :param activation: str, the type of activation function
        :param loss_type: str, the type of loss functions
        :param cost_type: str, the type of cost matrix for calculating optimal transport
            The length of the list is the number of modalities of the model
            Each element of the list is the number of event categories for each modality
        """
        super(HawkesProcessModel_OT, self).__init__(num_type, mu_dict, loss_type, use_cuda)
        self.model_name = 'A Hawkes Process'
        # self.num_type = num_type
        self.activation = activation
        exogenousIntensity = getattr(model.ExogenousIntensityFamily, mu_dict['model_name'])
        endogenousImpacts = getattr(model.EndogenousImpactFamily, alpha_dict['model_name'])
        decayKernel = getattr(model.DecayKernelFamily, kernel_dict['model_name'])

        mu_model = exogenousIntensity(num_type, mu_dict['parameter_set'])
        kernel_para = kernel_dict['parameter_set'].to(self.device)
        kernel_model = decayKernel(kernel_para)
        alpha_model = endogenousImpacts(num_type, kernel_model, alpha_dict['parameter_set'])

        self.lambda_model = HawkesProcessIntensity(mu_model, alpha_model, self.activation)
        self.print_info()
        self.dgw = GromovWassersteinDiscrepancy(loss_type=cost_type)
        self.dw = WassersteinDiscrepancy(loss_type=cost_type)

    def plot_exogenous(self, sample_dict, output_name: str = None):
        intensity = self.lambda_model.exogenous_intensity.intensity(sample_dict)
        self.lambda_model.exogenous_intensity.plot_and_save(intensity, output_name)

    def plot_causality(self, sample_dict, output_name: str = None):
        infectivity = self.lambda_model.endogenous_intensity.granger_causality(sample_dict)
        self.lambda_model.endogenous_intensity.plot_and_save(infectivity, output_name)

    def fit_ot(self, dataloader, optimizer, epochs: int,
               trans: torch.Tensor, mu_t: torch.Tensor, A_t: torch.Tensor, p_s: torch.Tensor, p_t: torch.Tensor,
               sample_dict1, sample_dict2, gamma, alpha,
               scheduler=None, sparsity: float=None, nonnegative=None,
               use_cuda: bool=False, validation_set=None):
        """
        Learn parameters of a generalized Hawkes process given observed sequences
        :param dataloader: a pytorch batch-based data loader
        :param optimizer: the sgd optimization method defined by PyTorch
        :param epochs: the number of training epochs
        :param trans: fixed optimal transport
        :param mu_t: base intensity of target Hawkes process
        :param A_t: infectivity of target Hawkes process
        :param p_s: the distribution of event types in source Hawkes process
        :param p_t: the distribution of event types in target Hawkes process
        :param scheduler: the method adjusting the learning rate of SGD defined by PyTorch
        :param sparsity: None or a float weight of L1 regularizer
        :param nonnegative: None or a float lower bound, typically the lower bound = 0
        :param use_cuda: use cuda (true) or not (false)
        :param validation_set: None or a validation dataloader
        """
        device = torch.device('cuda:0' if use_cuda else 'cpu')
        self.lambda_model.to(device)
        best_model = None
        self.lambda_model.train()

        if nonnegative is not None:
            clipper = LowerBoundClipper(nonnegative)

        Cs = torch.LongTensor(list(range(len(dataloader.dataset.database['type2idx']))))
        Cs = Cs.view(-1, 1)
        Cs = Cs.to(device)

        if dataloader.dataset.database['event_features'] is not None:
            all_event_feature = torch.from_numpy(dataloader.dataset.database['event_features'])
            FCs = all_event_feature.type(torch.FloatTensor)
            FCs = torch.t(FCs)    # (num_type, dim_features)
            FCs = FCs.to(device)
        else:
            FCs = None

        if validation_set is not None:
            validation_loss = self.validation(validation_set, use_cuda)
            logger.info('In the beginning, validation loss per event: {:.6f}.\n'.format(validation_loss))
            best_loss = validation_loss
        else:
            best_loss = np.inf

        for epoch in range(epochs):
            if scheduler is not None:
                scheduler.step()
            start = time.time()
            for batch_idx, samples in enumerate(dataloader):
                ci, batch_dict = samples2dict(samples, device, Cs, FCs)
                optimizer.zero_grad()
                lambda_t, Lambda_t = self.lambda_model(batch_dict)
                loss = self.loss_function(lambda_t, Lambda_t, ci) / lambda_t.size(0)
                reg = 0
                if sparsity is not None:
                    for parameter in self.lambda_model.parameters():
                        reg += sparsity * torch.sum(torch.abs(parameter))

                base_intensity = self.lambda_model.exogenous_intensity.intensity(sample_dict1)
                infectivity = self.lambda_model.endogenous_intensity.granger_causality(sample_dict2).squeeze(2)
                d_gw = self.dgw(infectivity, A_t, trans, p_s, p_t)
                d_w = self.dw(base_intensity, mu_t, trans, p_s, p_t)
                loss_total = loss + reg + gamma * (alpha*d_w + (1-alpha)*d_gw)
                loss_total.backward()
                optimizer.step()
                if nonnegative is not None:
                    self.lambda_model.apply(clipper)

                # display training processes
                if batch_idx % 100 == 0:
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                        epoch, batch_idx * ci.size(0), len(dataloader.dataset), 100. * batch_idx / len(dataloader)))
                    if sparsity is not None:
                        logger.info('Loss per event: {:.6f}, Regularizer: {:.6f} Time={:.2f}sec'.format(
                            loss.data, reg.data, time.time() - start))
                    else:
                        logger.info('Loss per event: {:.6f}, Regularizer: {:.6f} Time={:.2f}sec'.format(
                            loss.data, 0, time.time() - start))

            if validation_set is not None:
                validation_loss = self.validation(validation_set, use_cuda)
                logger.info('After Epoch: {}, validation loss per event: {:.6f}.\n'.format(epoch, validation_loss))
                if validation_loss < best_loss:
                    best_model = copy.deepcopy(self.lambda_model)
                    best_loss = validation_loss

        if best_model is not None:
            self.lambda_model = copy.deepcopy(best_model)


def fused_gromov_wasserstein_discrepancy(p_s, p_t, A_s, A_t, mu_s, mu_t, hyperpara_dict):
    """
    Learning optimal transport from source to target domain

    Args:
        cost_s: (Ns, Ns) matrix representing the relationships among source entities
        cost_t: (Nt, Nt) matrix representing the relationships among target entities
        cost_mutual: (Ns, Nt) matrix representing the prior of proposed optimal transport
        mu_s: (Ns, 1) vector representing marginal probability of source entities
        mu_t: (Nt, 1) vector representing marginal probability of target entities
        hyperpara_dict: a dictionary of hyperparameters
            dict = {epochs: the number of epochs,
                    batch_size: batch size,
                    use_cuda: use cuda or not,
                    strategy: hard or soft,
                    beta: the weight of proximal term
                    outer_iter: the outer iteration of ipot
                    inner_iter: the inner iteration of sinkhorn
                    prior: True or False
                    }

    Returns:

    """
    ns = p_s.size(0)
    nt = p_t.size(0)
    trans = torch.matmul(p_s, torch.t(p_t))
    a = mu_s.sum().repeat(ns, 1)
    a /= a.sum()
    b = 0
    beta = hyperpara_dict['beta']

    if hyperpara_dict['loss_type'] == 'L2':
        fs = (mu_s ** 2).repeat(1, nt)
        ft = torch.t(mu_t ** 2).repeat(ns, 1)
        cost_st = fs + ft
        cost_mu = cost_st - 2 * torch.matmul(mu_s, torch.t(mu_t))

        # f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = 2b
        # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
        # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
        f1_st = torch.matmul(A_s ** 2, p_s).repeat(1, nt)
        f2_st = torch.matmul(torch.t(p_t), torch.t(A_t ** 2)).repeat(ns, 1)
        cost_st = f1_st + f2_st
        for t in range(hyperpara_dict['outer_iteration']):
            cost_A = cost_st - 2 * torch.matmul(torch.matmul(A_s, trans), torch.t(A_t))
            cost = hyperpara_dict['alpha'] * cost_mu + (1 - hyperpara_dict['alpha']) * cost_A
            if hyperpara_dict['ot_method'] == 'proximal':
                kernel = torch.exp(-cost / beta) * trans
            else:
                kernel = torch.exp(-cost / beta)
            for l in range(hyperpara_dict['inner_iteration']):
                b = p_t / torch.matmul(torch.t(kernel), a)
                a = p_s / torch.matmul(kernel, b)
                # print((b**2).sum())
                # print((a**2).sum())
                # print((b**2).sum()*(a**2).sum())
            trans = torch.matmul(torch.matmul(torch.diag(a[:, 0]), kernel), torch.diag(b[:, 0]))
            if t % 100 == 0:
                print('sinkhorn iter {}/{}'.format(t, hyperpara_dict['outer_iteration']))
        cost_A = cost_st - 2 * torch.matmul(torch.matmul(A_s, trans), torch.t(A_t))

    else:
        fs = (mu_s ** 2).repeat(1, nt)
        ft = torch.t(mu_t ** 2).repeat(ns, 1)
        cost_mu = fs * torch.log(fs / (ft + 1e-5)) - fs + ft

        # f1(a) = a*log(a) - a, f2(b) = b, h1(a) = a, h2(b) = log(b)
        # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
        # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
        f1_st = torch.matmul(A_s * torch.log(A_s + 1e-5) - A_s, p_s).repeat(1, nt)
        f2_st = torch.matmul(torch.t(p_t), torch.t(A_t)).repeat(ns, 1)
        cost_st = f1_st + f2_st
        for t in range(hyperpara_dict['outer_iteration']):
            cost_A = cost_st - torch.matmul(torch.matmul(A_s, trans), torch.t(torch.log(A_t + 1e-5)))
            cost = hyperpara_dict['alpha'] * cost_mu + (1 - hyperpara_dict['alpha']) * cost_A
            if hyperpara_dict['ot_method'] == 'proximal':
                kernel = torch.exp(-cost / beta) * trans
            else:
                kernel = torch.exp(-cost / beta)
            for l in range(hyperpara_dict['inner_iteration']):
                b = p_t / torch.matmul(torch.t(kernel), a)
                a = p_s / torch.matmul(kernel, b)
            trans = torch.matmul(torch.matmul(torch.diag(a[:, 0]), kernel), torch.diag(b[:, 0]))
        cost_A = cost_st - torch.matmul(torch.matmul(A_s, trans), torch.t(torch.log(A_t + 1e-5)))

    cost = hyperpara_dict['alpha'] * cost_mu + (1 - hyperpara_dict['alpha']) * cost_A
    d_fgw = (cost * trans).sum()
    return trans, d_fgw


def wasserstein_discrepancy(p_s, p_t, mu_s, mu_t, hyperpara_dict):
    """
    Learning optimal transport from source to target domain

    Args:
        cost_s: (Ns, Ns) matrix representing the relationships among source entities
        cost_t: (Nt, Nt) matrix representing the relationships among target entities
        cost_mutual: (Ns, Nt) matrix representing the prior of proposed optimal transport
        mu_s: (Ns, 1) vector representing marginal probability of source entities
        mu_t: (Nt, 1) vector representing marginal probability of target entities
        hyperpara_dict: a dictionary of hyperparameters
            dict = {epochs: the number of epochs,
                    batch_size: batch size,
                    use_cuda: use cuda or not,
                    strategy: hard or soft,
                    beta: the weight of proximal term
                    outer_iter: the outer iteration of ipot
                    inner_iter: the inner iteration of sinkhorn
                    prior: True or False
                    }

    Returns:

    """
    ns = p_s.size(0)
    nt = p_t.size(0)
    trans = torch.matmul(p_s, torch.t(p_t))
    a = mu_s.sum().repeat(ns, 1)
    a /= a.sum()
    b = 0
    beta = hyperpara_dict['beta']

    if hyperpara_dict['loss_type'] == 'L2':
        fs = (mu_s ** 2).repeat(1, nt)
        ft = torch.t(mu_t ** 2).repeat(ns, 1)
        cost_st = fs + ft
        cost_mu = cost_st - 2 * torch.matmul(mu_s, torch.t(mu_t))

    else:
        fs = (mu_s ** 2).repeat(1, nt)
        ft = torch.t(mu_t ** 2).repeat(ns, 1)
        cost_mu = fs * torch.log(fs / (ft + 1e-5)) - fs + ft

    for t in range(hyperpara_dict['outer_iteration']):
        if hyperpara_dict['ot_method'] == 'proximal':
            kernel = torch.exp(-cost_mu / beta) * trans
        else:
            kernel = torch.exp(-cost_mu / beta)
        for l in range(hyperpara_dict['inner_iteration']):
            b = p_t / torch.matmul(torch.t(kernel), a)
            a = p_s / torch.matmul(kernel, b)
        trans = torch.matmul(torch.matmul(torch.diag(a[:, 0]), kernel), torch.diag(b[:, 0]))

    d_w = (cost_mu * trans).sum()
    return trans, d_w
