"""
This script contains a parent class of exogenous intensity function mu(t).
"""

import torch
import torch.nn as nn
from typing import Dict
from dev.util import logger
import matplotlib.pyplot as plt


class BasicExogenousIntensity(nn.Module):
    """
    The parent class of exogenous intensity function mu(t), which actually a constant exogenous intensity
    """
    def __init__(self, num_type: int):
        """
        Initialize exogenous intensity function: mu(t) = mu, mu in R^{C+1}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        """
        super(BasicExogenousIntensity, self).__init__()
        self.exogenous_intensity_type = 'constant'
        self.activation = 'identity'

        self.num_type = num_type
        self.dim_embedding = 1
        self.emb = nn.Embedding(self.num_type, self.dim_embedding, padding_idx=0)
        self.emb.weight = nn.Parameter(
            torch.cat([torch.zeros(1, self.dim_embedding),
                       torch.FloatTensor(self.num_type - 1, self.dim_embedding).uniform_(0.01 / self.dim_embedding,
                                                                                         1 / self.dim_embedding)],
                      dim=0))

    def print_info(self):
        """
        Print basic information of the exogenous intensity function.
        """
        logger.info('Exogenous intensity function: mu(t) = {}.'.format(self.exogenous_intensity_type))
        logger.info('The number of event types = {}.'.format(self.num_type))

    def forward(self, sample_dict: Dict):
        """
        Calculate
        1) mu_{c_i} for c_i in "events";
        2) int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            }
        :return:
            mu_c: (batch_size, 1) FloatTensor represents mu_{c_i};
            mU: (batch_size, num_type) FloatTensor represents int_{0}^{dt} mu_c(s)ds
        """
        mu_c = self.intensity(sample_dict)
        mU = self.expect_counts(sample_dict)
        return mu_c, mU

    def intensity(self, sample_dict: Dict):
        """
        Calculate intensity mu_{c_i} for c_i in "events";

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            }
        :return:
            mu_c: (batch_size, 1) FloatTensor represents mu_{c_i};
        """
        events = sample_dict['ci']  # (batch_size, 1)
        mu_c = self.emb(events)     # (batch_size, 1, 1)
        mu_c = mu_c.squeeze(1)      # (batch_size, 1)
        return mu_c

    def expect_counts(self, sample_dict: Dict):
        """
        Calculate expected number of events in dts
        int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            }
        :return:
            mU: (batch_size, num_type) FloatTensor represents int_{0}^{dt} mu_c(s)ds
        """
        dts = sample_dict['ti'] - sample_dict['tjs'][:, -1].view(-1, 1)
        all_types = sample_dict['Cs']  # (num_type, 1)
        mu_all = self.emb(all_types)   # (num_type, 1, 1)
        mu_all = mu_all.squeeze(1)     # (num_type, 1)
        mU = torch.matmul(dts, torch.t(mu_all))  # (batch_size, num_type)
        return mU

    def plot_and_save(self, mu_all: torch.Tensor, output_name: str = None):
        """
        Plot the stem plot of exogenous intensity functions for all event types
        Args:
        :param mu_all: a (num_type, 1) FloatTensor containing all exogenous intensity functions
        :param output_name: the name of the output png file
        """
        mu_all = mu_all.squeeze(1)  # (C,)
        mu_all = mu_all.data.numpy()

        plt.figure(figsize=(5, 5))
        plt.stem(range(mu_all.shape[0]), mu_all, '-', c='r')
        plt.ylabel('Exogenous intensity')
        plt.xlabel('Index of event type')
        if output_name is None:
            plt.savefig('exogenous_intensity.png')
        else:
            plt.savefig(output_name)
        plt.close("all")
        logger.info("Done!")




