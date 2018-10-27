"""
The classes of specific decay kernels
"""

import copy
from dev.util import logger
import torch
from typing import Optional
import numpy as np
from model.DecayKernel import BasicDecayKernel


class ExponentialKernel(BasicDecayKernel):
    """
    The class of exponential kernel, which is inherited from the parent class directly.
    """
    pass


class RayleighKernel(BasicDecayKernel):
    """
    The class of Rayleigh kernel
    """
    def __init__(self, parameters: torch.Tensor):
        """
        Initialize decay functions
            rayleigh,
                g(t) = wt * exp(-wt^2/2) if t>0
        :param parameters: the parameters related to decay kernel
                (1, 1) Tensor for bandwidth
        """
        super(RayleighKernel, self).__init__(parameters)
        self.kernel_type = 'Rayleigh'

    def values(self, dt: torch.Tensor) -> torch.Tensor:
        """
        Calculate decay kernel's value at time 'dt'
        :param dt: a 2D Tensor containing the time intervals between current event and historical ones
        :return:
            gt: a Tensor containing decay kernel's values at different time.
        """
        sigma2 = self.parameters[0, 0]
        w = 1 / sigma2
        gt = (w * dt) * torch.exp(-0.5 * w * (dt ** 2))
        gt2 = gt.view(gt.size(0), gt.size(1), 1)
        # gt2 = np.zeros((dt.shape[0], dt.shape[1], 1))
        # gt2[:, :, 0] = gt
        return gt2

    def integrations(self, t_stop: torch.Tensor, t_start: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate the integrations of decay kernel in the interval [t_start, t_stop]
        :param t_stop: a 2D Tensor containing stop timestamps
        :param t_start: a 2D Tensor containing start timestamps, if it is None, it means t_start = 0
        :return:
            gt: a Tensor containing decay kernel's integration values in the interval [t_start, t_stop].
        """
        if t_start is None:
            t_start = 0 * t_stop

        if t_start.size() != t_stop.size():
            logger.warning(f"The t_start does not have the same shape with t_stop, we set t_start to all zeros")
            t_start = 0 * t_stop

        sigma2 = self.parameters[0, 0]
        w = 1 / sigma2
        gt_start = torch.exp(-0.5 * w * (t_start ** 2))
        gt_stop = torch.exp(-0.5 * w * (t_stop ** 2))

        gt_d = gt_stop - gt_start
        gt = - gt_d.view(gt_d.size(0), gt_d.size(1), 1)
        # gt = np.zeros((gt_d.shape[0], gt_d.shape[1], 1))
        # gt[:, :, 0] = - gt_d
        return gt


class GaussianKernel(BasicDecayKernel):
    """
    The class of Gaussian kernel
    """
    def __init__(self, parameters: torch.Tensor):
        """
        Initialize decay functions
            rayleigh,
                g(t) = w * exp(-wt^2/2) if t>0
        :param parameters: the parameters related to decay kernel
                (1, 1) Tensor for bandwidth
        """
        super(GaussianKernel, self).__init__(parameters)
        self.kernel_type = 'Gaussian'

    def values(self, dt: torch.Tensor) -> torch.Tensor:
        """
        Calculate decay kernel's value at time 'dt'
        :param dt: a 2D Tensor containing the time intervals between current event and historical ones
        :return:
            gt: a Tensor containing decay kernel's values at different time.
        """
        sigma2 = self.parameters[0, 0]
        gt = 1 / torch.sqrt(2 * np.pi * sigma2) * torch.exp(-0.5 * (dt ** 2) / sigma2)

        # gt2 = np.zeros((dt.shape[0], dt.shape[1], 1))
        gt2 = gt.view(gt.size(0), gt.size(1), 1)
        gt2[:, :, 0] = gt
        return gt2

    def integrations(self, t_stop: torch.Tensor, t_start: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate the integrations of decay kernel in the interval [t_start, t_stop]
        :param t_stop: a 2D Tensor containing stop timestamps
        :param t_start: a 2D Tensor containing start timestamps, if it is None, it means t_start = 0
        :return:
            gt: a Tensor containing decay kernel's integration values in the interval [t_start, t_stop].
        """
        if t_start is None:
            t_start = 0 * t_stop

        if t_start.size() != t_stop.size():
            logger.warning(f"The t_start does not have the same shape with t_stop, we set t_start to all zeros")
            t_start = 0 * t_stop

        sigma2 = self.parameters[0, 0]
        gt_start = 0.5 * (1 + torch.erf(t_start / (torch.sqrt(2 * sigma2))))
        gt_stop = 0.5 * (1 + torch.erf(t_stop / (torch.sqrt(2 * sigma2))))

        gt_d = gt_stop - gt_start
        gt = gt_d.view(gt_d.size(0), gt_d.size(1), 1)
        # gt = np.zeros((gt_d.shape[0], gt_d.shape[1], 1))
        # gt[:, :, 0] = gt_d
        return gt


class PowerlawKernel(BasicDecayKernel):
    """
    The class of powerlaw kernel.
    """
    def __init__(self, parameters: torch.Tensor):
        """
        Initialize decay functions
            powerlaw,
                g(t) = (w-1)*delay^(w-1)*t^(-w) if t>=delay, = (w-1)/delay if t<delay
        :param parameters: the parameters related to decay kernel
                (2, 1) tensor for decay and bandwidth
        """
        super(PowerlawKernel, self).__init__(parameters)
        self.kernel_type = 'Power-Law'

    def values(self, dt: torch.Tensor) -> torch.Tensor:
        """
        Calculate decay kernel's value at time 'dt'
        :param dt: a 2D Tensor containing the time intervals between current event and historical ones
        :return:
            gt: a Tensor containing decay kernel's values at different time.
        """
        delay = self.parameters[0, 0]
        bandwidth = self.parameters[1, 0]
        coefficient = (bandwidth - 1) * (delay ** (bandwidth - 1))
        dt2 = copy.deepcopy(dt)
        dt2[dt2 == 0] = 1e-7
        gt = coefficient * (dt2 ** (-bandwidth))
        gt[dt2 - delay < 0] = (bandwidth - 1) / delay
        gt2 = gt.view(gt.size(0), gt.size(1), 1)
        # gt2 = np.zeros((dt.shape[0], dt.shape[1], 1))
        # gt2[:, :, 0] = gt
        return gt2

    def integrations(self, t_stop: torch.Tensor, t_start: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate the integrations of decay kernel in the interval [t_start, t_stop]
        :param t_stop: a 2D Tensor containing stop timestamps
        :param t_start: a 2D Tensor containing start timestamps, if it is None, it means t_start = 0
        :return:
            gt: a Tensor containing decay kernel's integration values in the interval [t_start, t_stop].
        """
        if t_start is None:
            t_start = 0 * t_stop

        if t_start.size() != t_stop.size():
            logger.warning(f"The t_start does not have the same shape with t_stop, we set t_start to all zeros")
            t_start = 0 * t_stop

        delay = self.parameters[0, 0]
        bandwidth = self.parameters[1, 0]

        # condition 1
        dt_start1 = copy.deepcopy(t_start)
        dt_start1[t_start > delay] = 0
        gt_start1 = dt_start1 * (bandwidth - 1) / delay
        # condition 2
        dt_start2 = copy.deepcopy(t_start)
        dt_start2[t_start <= delay] = delay
        gt_start2 = 1 - (delay/dt_start2)**(bandwidth - 1) + (bandwidth - 1)
        gt_start2[t_start <= delay] = 0
        gt_start = gt_start1 + gt_start2

        # condition 1
        dt_stop1 = copy.deepcopy(t_stop)
        dt_stop1[t_stop > delay] = 0
        gt_stop1 = dt_stop1 * (bandwidth - 1) / delay
        # condition 2
        dt_stop2 = copy.deepcopy(t_stop)
        dt_stop2[t_stop <= delay] = delay
        gt_stop2 = 1 - (delay/dt_stop2)**(bandwidth - 1) + (bandwidth - 1)
        gt_stop2[t_stop <= delay] = 0
        gt_stop = gt_stop1 + gt_stop2

        gt_d = gt_stop - gt_start
        gt = gt_d.view(gt_d.size(0), gt_d.size(1), 1)
        # gt = np.zeros((gt_d.shape[0], gt_d.shape[1], 1))
        # gt[:, :, 0] = gt_d
        return gt


class GateKernel(BasicDecayKernel):
    """
    The class of gate kernel.
    """

    def __init__(self, parameters: torch.Tensor):
        """
        Initialize decay functions
            gate,
                g(t) = 1/a, if t in [delay, delay + a], = 0 otherwise.
        :param parameters: the parameters related to decay kernel
                (2, 1) tensor for decay and bandwidth
        """
        super(GateKernel, self).__init__(parameters)
        self.kernel_type = 'Gate'

    def values(self, dt: torch.Tensor) -> torch.Tensor:
        """
        Calculate decay kernel's value at time 'dt'
        :param dt: a 2D Tensor containing the time intervals between current event and historical ones
        :return:
            gt: a Tensor containing decay kernel's values at different time.
        """
        delay = self.parameters[0, 0]
        bandwidth = self.parameters[1, 0]
        w = 1 / bandwidth
        gt = w.repeat(dt.size(0), dt.size(1))
        gt[dt - delay < 0] = 0
        gt[dt - delay - bandwidth > 0] = 0
        gt2 = gt.view(gt.size(0), gt.size(1), 1)
        # gt2 = np.zeros((dt.shape[0], dt.shape[1], 1))
        # gt2[:, :, 0] = gt
        return gt2

    def integrations(self, t_stop: torch.Tensor, t_start: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate the integrations of decay kernel in the interval [t_start, t_stop]
        :param t_stop: a 2D Tensor containing stop timestamps
        :param t_start: a 2D Tensor containing start timestamps, if it is None, it means t_start = 0
        :return:
            gt: a Tensor containing decay kernel's integration values in the interval [t_start, t_stop].
        """
        if t_start is None:
            t_start = 0 * t_stop

        if t_start.size() != t_stop.size():
            logger.warning(f"The t_start does not have the same shape with t_stop, we set t_start to all zeros")
            t_start = 0 * t_stop

        delay = self.parameters[0, 0]
        bandwidth = self.parameters[1, 0]
        w = 1 / bandwidth
        gt_start = w * (t_start - delay)
        gt_start[gt_start < 0] = 0
        gt_start[gt_start > 1] = 1

        gt_stop = w * (t_stop - delay)
        gt_stop[gt_stop < 0] = 0
        gt_stop[gt_stop > 1] = 1

        gt_d = gt_stop - gt_start
        gt = gt_d.view(gt_d.size(0), gt_d.size(1), 1)
        # gt = np.zeros((gt_d.shape[0], gt_d.shape[1], 1))
        # gt[:, :, 0] = gt_d
        return gt


class MultiGaussKernel(BasicDecayKernel):
    """
    The class of multi-gaussian kernel.
    """

    def __init__(self, parameters: torch.Tensor):
        """
        Initialize decay functions
            multigauss.
                g(t) = sum_i 1/sqrt(2pi*sigma_i^2) * exp(-(t-mu_i)^2/(2*sigma_i^2))
        :param parameters: the parameters related to decay kernel
                (2, N) tensor for landmarks and bandwidths
        """
        super(MultiGaussKernel, self).__init__(parameters)
        self.kernel_type = 'Multi-Gauss'

    def values(self, dt: torch.Tensor) -> torch.Tensor:
        """
        Calculate decay kernel's value at time 'dt'
        :param dt: a 2D Tensor containing the time intervals between current event and historical ones
        :return:
            gt: a Tensor containing decay kernel's values at different time.
        """
        landmarks = self.parameters[0, :]
        sigma2 = self.parameters[1, :]
        gt = 0 * dt.unsqueeze(2).repeat(1, 1, self.parameters.size(1))
        # gt = torch.zeros(dt.size(0), dt.size(1), self.parameters.size(1))
        for i in range(self.parameters.size(1)):
            gt[:, :, i] = 1 / torch.sqrt(2 * np.pi * sigma2[i]) * \
                          torch.exp(-0.5 * ((dt - landmarks[i]) ** 2) / sigma2[i])
        return gt

    def integrations(self, t_stop: torch.Tensor, t_start: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate the integrations of decay kernel in the interval [t_start, t_stop]
        :param t_stop: a 2D Tensor containing stop timestamps
        :param t_start: a 2D Tensor containing start timestamps, if it is None, it means t_start = 0
        :return:
            gt: a Tensor containing decay kernel's integration values in the interval [t_start, t_stop].
        """
        if t_start is None:
            t_start = 0 * t_stop

        if t_start.size() != t_stop.size():
            logger.warning(f"The t_start does not have the same shape with t_stop, we set t_start to all zeros")
            t_start = 0 * t_stop

        landmarks = self.parameters[0, :]
        sigma2 = self.parameters[1, :]
        gt = 0 * t_stop.unsqueeze(2).repeat(1, 1, self.parameters.size(1))
        # gt = torch.zeros(t_stop.size(0), t_stop.size(1), self.parameters.size(1))
        for i in range(self.parameters.shape[1]):
            gt_start = 0.5 * (1 + torch.erf((t_start - landmarks[i]) / (torch.sqrt(2 * sigma2[i]))))
            gt_stop = 0.5 * (1 + torch.erf((t_stop - landmarks[i]) / (torch.sqrt(2 * sigma2[i]))))
            gt[:, :, i] = gt_stop - gt_start
        return gt
