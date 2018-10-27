"""
This script contains a parent class of decay functions, which contains basic functions of following decay kernels:

1) exponential kernel
2) rayleigh kernel
3) gaussian kernel
4) powerlaw kernel
5) gate kernel
6) multi-gaussian kernel

Written by Hongteng Xu, on Sep. 4, 2018
"""

from dev.util import logger
from typing import Optional
import numpy as np
import torch
import matplotlib.pyplot as plt


class BasicDecayKernel(object):
    """
    The parent class of decay functions, which actually an exponential kernel
    """
    def __init__(self, parameters: torch.Tensor):
        """
        Initialize decay functions
        exponential
                g(t) = w * exp(-w(t-delay)) if t>=delay, = 0 if t<delay
        :param parameters: the parameters related to decay kernel
                (2, 1) array for decay and bandwidth
        """
        self.kernel_type = 'Exponential'
        self.parameters = parameters

    def print_info(self):
        """
        Print basic information of the kernel model.
        """
        logger.info('The type of decay kernel: {}.'.format(self.kernel_type))
        logger.info('The number of basis = {}.'.format(self.parameters.size(1)))
        # logger.info('The number of basis = {}.'.format(self.parameters.shape[1]))

    def values(self, dt: torch.Tensor) -> torch.Tensor:
        """
        Calculate decay kernel's value at time 'dt'
        :param dt: a 2D Tensor containing the time intervals between current event and historical ones
        :return:
            gt: a Tensor containing decay kernel's values at different time.
        """
        delay = self.parameters[0, 0]
        bandwidth = self.parameters[1, 0]
        # w = np.sqrt(1 / bandwidth)
        w = torch.sqrt(1 / bandwidth)  # ** 0.5

        dt2 = dt - delay
        # gt = w * np.exp(-w * dt2)
        gt = w * torch.exp(-w * dt2)
        gt[dt2 < 0] = 0
        # gt2 = np.zeros((dt.shape[0], dt.shape[1], 1))
        # gt2[:, :, 0] = gt
        gt2 = gt.view(gt.size(0), gt.size(1), 1)
        return gt2

    def integrations(self, t_stop: torch.Tensor, t_start: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate the integrations of decay kernel in the interval [t_start, t_stop]
        :param t_stop: a 2D Tensor containing stop timestamps
        :param t_start: a 2D Tensor containing start timestamps, if it is None, it means t_start = 0
        :return:
            gt: the ndarray containing decay kernel's integration values in the interval [t_start, t_stop].
        """
        if t_start is None:
            t_start = 0 * t_stop

        if t_start.size() != t_stop.size():
            logger.warning(f"The t_start does not have the same shape with t_stop, we set t_start to all zeros")
            t_start = 0 * t_stop

        delay = self.parameters[0, 0]
        bandwidth = self.parameters[1, 0]
        # w = np.sqrt(1 / bandwidth)
        w = torch.sqrt(1 / bandwidth)
        # gt_start = np.exp(-w * (t_start - delay))
        gt_start = (-w * (t_start - delay)).exp()
        gt_start[gt_start > 1] = 1
        # gt_stop = np.exp(-w * (t_stop - delay))
        gt_stop = (-w * (t_stop - delay)).exp()
        gt_stop[gt_stop > 1] = 1

        gt_d = gt_stop - gt_start
        # gt = np.zeros((gt_d.shape[0], gt_d.shape[1], 1))
        # gt[:, :, 0] = -gt_d
        gt = -gt_d.view(gt_d.size(0), gt_d.size(1), 1)
        return gt

    def plot_and_save(self, t_stop: float = 5.0, output_name: str = None):
        """
        Plot decay function and its integration and save the figure as a png file
        Args:
            t_stop (float): the end of timestamp
            output_name (str): the name of the output png file
        """

        dt = np.arange(0.0, t_stop, 0.01)
        dt = np.tile(dt, (1, 1))
        dt = torch.from_numpy(dt)
        dt = dt.type(torch.FloatTensor)
        gt = self.values(dt)
        # t_start = torch.zeros(dt.size())
        igt = self.integrations(dt)
        # print(gt.shape)

        plt.figure(figsize=(5, 5))
        for k in range(gt.shape[2]):
            plt.plot(dt[0, :].cpu().numpy(), gt[0, :, k].cpu().numpy(), label='g_{}(t)'.format(k), c='r')
            plt.plot(dt[0, :].cpu().numpy(), igt[0, :, k].cpu().numpy(), label='G_{}(t)'.format(k), c='b')
        leg = plt.legend(loc='upper left', ncol=1, shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)
        plt.title('{} decay kernel and its integration'.format(self.kernel_type))
        if output_name is None:
            plt.savefig('{}_decay_kernel.png'.format(self.kernel_type))
        else:
            plt.savefig(output_name)
        plt.close("all")
        logger.info("Done!")
