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

import torch.nn as nn
from dev.util import logger
from model.OtherLayers import Identity
from model.PointProcess import PointProcessModel
import model.ExogenousIntensityFamily
import model.EndogenousImpactFamily
import model.DecayKernelFamily


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
