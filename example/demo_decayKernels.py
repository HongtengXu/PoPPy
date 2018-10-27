"""
An example showing different decay kernels we implemented
"""

import dev.util as util
import numpy as np
import model.DecayKernelFamily as dKernel
import torch

# exponential kernel
parameters = np.ones((2, 1))
parameters = torch.from_numpy(parameters)
parameters = parameters.type(torch.FloatTensor)
kernel = dKernel.ExponentialKernel(parameters)
kernel.print_info()
kernel.plot_and_save(output_name='{}/{}/kernel_exp.png'.format(util.POPPY_PATH, util.OUTPUT_DIR))

# Rayleigh kernel
parameters = 2*np.ones((1, 1))
parameters = torch.from_numpy(parameters)
parameters = parameters.type(torch.FloatTensor)
kernel = dKernel.RayleighKernel(parameters)
kernel.print_info()
kernel.plot_and_save(output_name='{}/{}/kernel_rayleigh.png'.format(util.POPPY_PATH, util.OUTPUT_DIR))

# Gaussian kernel
parameters = 2*np.ones((1, 1))
parameters = torch.from_numpy(parameters)
parameters = parameters.type(torch.FloatTensor)
kernel = dKernel.GaussianKernel(parameters)
kernel.print_info()
kernel.plot_and_save(output_name='{}/{}/kernel_gauss.png'.format(util.POPPY_PATH, util.OUTPUT_DIR))

# PowerLaw kernel
parameters = 2*np.ones((2, 1))
parameters = torch.from_numpy(parameters)
parameters = parameters.type(torch.FloatTensor)
kernel = dKernel.PowerlawKernel(parameters)
kernel.print_info()
kernel.plot_and_save(output_name='{}/{}/kernel_power.png'.format(util.POPPY_PATH, util.OUTPUT_DIR))

# Gate kernel
parameters = 2*np.ones((2, 1))
parameters = torch.from_numpy(parameters)
parameters = parameters.type(torch.FloatTensor)
kernel = dKernel.GateKernel(parameters)
kernel.print_info()
kernel.plot_and_save(output_name='{}/{}/kernel_gate.png'.format(util.POPPY_PATH, util.OUTPUT_DIR))

# MultiGaussian kernel
parameters = np.arange(0, 4, 1)
parameters = np.tile(parameters, (2, 1))
parameters[1, :] += 1
parameters = torch.from_numpy(parameters)
parameters = parameters.type(torch.FloatTensor)
kernel = dKernel.MultiGaussKernel(parameters)
kernel.print_info()
kernel.plot_and_save(output_name='{}/{}/kernel_multi-gauss.png'.format(util.POPPY_PATH, util.OUTPUT_DIR))
