"""
This script contains the patent class of endogenous impact function
"""

import torch
import torch.nn as nn
from typing import Dict
from dev.util import logger
import matplotlib.pyplot as plt


class BasicEndogenousImpact(nn.Module):
    """
    The parent class of endogenous impact functions sum_i phi_{kk_i}(t-t_i) for k = 1,...,C,
    which actually a simple endogenous impact with phi_{kk'}(t) = sum_{m} a_{kk'm} kernel_m(t)
    """

    def __init__(self, num_type: int, kernel):
        """
        Initialize endogenous impact: phi_{kk'}(t) = sum_{m} a_{kk'm} kernel_m(t),
        for m = 1, ..., M, A_m = [a_{kk'm}] in R^{C*C+1}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        :param kernel: an instance of a decay kernel class in "DecayKernelFamily"
        """
        super(BasicEndogenousImpact, self).__init__()
        self.decay_kernel = kernel
        self.num_base = self.decay_kernel.parameters.shape[1]
        self.endogenous_impact_type = "sum_m a_(kk'm) * kernel_m(t)"
        self.num_type = num_type
        self.dim_embedding = num_type
        for m in range(self.num_base):
            # emb = nn.Embedding(self.num_type, self.dim_embedding, padding_idx=0)
            # emb.weight = nn.Parameter(
            #     torch.cat([torch.zeros(self.num_type, 1),
            #                torch.FloatTensor(self.num_type, self.dim_embedding - 1).uniform_(0.01 / self.dim_embedding,
            #                                                                                  1 / self.dim_embedding)],
            #               dim=1))
            emb = nn.Embedding(self.num_type, self.dim_embedding)
            emb.weight = nn.Parameter(
                torch.FloatTensor(self.num_type, self.dim_embedding).uniform_(0.01 / self.dim_embedding,
                                                                              1 / self.dim_embedding))
            if m == 0:
                self.basis = nn.ModuleList([emb])
            else:
                self.basis.append(emb)

    def print_info(self):
        """
        Print basic information of the exogenous intensity function.
        """
        logger.info("Endogenous impact function: phi_(kk')(t) = {}.".format(self.endogenous_impact_type))
        logger.info('The number of event types = {}.'.format(self.num_type))
        self.decay_kernel.print_info()

    def intensity(self, sample_dict: Dict):
        """
        Calculate intensity of event
        phi_{c_i,c_j}(t_i - t_j) for c_i in "events";

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
            pHi: (batch_size, num_type) FloatTensor represents sum_{c} sum_{i in history} int_{start}^{stop} phi_cc_i(s)ds
        """
        event_time = sample_dict['ti']     # (batch_size, 1)
        history_time = sample_dict['tjs']  # (batch_size, memory_size)
        events = sample_dict['ci']         # (batch_size, 1)
        history = sample_dict['cjs']       # (batch_size, memory_size)
        dts = event_time.repeat(1, history_time.size(1)) - history_time  # (batch_size, memory_size)
        gt = self.decay_kernel.values(dts.numpy())
        gt = torch.from_numpy(gt)
        gt = gt.type(torch.FloatTensor)    # (batch_size, memory_size, num_base)

        phi_c = 0
        for m in range(self.num_base):
            A_cm = self.basis[m](events)                        # (batch_size, 1, dim_embedding)
            A_cm = A_cm.squeeze(1)                              # (batch_size, dim_embedding)
            A_cm = A_cm.gather(1, history)                      # (batch_size, memory_size)
            A_cm = A_cm.unsqueeze(1)                            # (batch_size, 1, memory_size)
            phi_c += torch.bmm(A_cm, gt[:, :, m].unsqueeze(2))  # (batch_size, 1, 1)
        phi_c = phi_c[:, :, 0]
        return phi_c

    def expect_counts(self, sample_dict: Dict):
        """
        Calculate the expected number of events in dts
        sum_i int_{0}^{dt_i} phi_cc_i(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
            pHi: (batch_size, num_type) FloatTensor represents sum_{c} sum_{i in history} int_{start}^{stop} phi_cc_i(s)ds
        """
        event_time = sample_dict['ti']     # (batch_size, 1)
        history_time = sample_dict['tjs']  # (batch_size, memory_size)
        history = sample_dict['cjs']       # (batch_size, memory_size)
        all_types = sample_dict['Cs']      # (num_type, 1)

        dts = event_time.repeat(1, history_time.size(1)) - history_time   # (batch_size, memory_size)
        t_start = history_time[:, -1].repeat(1, history_time.size(1)) - history_time  # (batch_size, memory_size)
        t_stop = dts                                                                  # (batch_size, memory_size)
        Gt = self.decay_kernel.integrations(t_stop.numpy(), t_start.numpy())
        Gt = torch.from_numpy(Gt)
        Gt = Gt.type(torch.FloatTensor)                                   # (batch_size, memory_size, num_base)
        history2 = history.unsqueeze(1).repeat(1, all_types.size(0), 1)   # (batch_size, num_type, memory_size)

        pHi = 0
        for m in range(self.num_base):
            A_all = self.basis[m](all_types)                      # (num_type, 1, dim_embedding)
            A_all = A_all.squeeze(1).unsqueeze(0)                 # (1, num_type, dim_embedding)
            A_all = A_all.repeat(Gt.size(0), 1, 1)                # (batch_size, num_type, dim_embedding)
            A_all = A_all.gather(2, history2)                     # (batch_size, num_type, memory_size)
            pHi += torch.bmm(A_all, Gt[:, :, m].unsqueeze(2))     # (batch_size, num_type, 1)
        pHi = pHi[:, :, 0]
        return pHi

    def granger_causality(self, sample_dict: Dict):
        """
        Calculate the granger causality among event types
        a_{cc'm}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            }
        :return:
            A_all: (num_type, num_type, num_base) FloatTensor represents a_{cc'm} in phi_{cc'}(t)
        """
        all_types = sample_dict['Cs']  # (num_type, 1)
        A_all = 0
        for m in range(self.num_base):
            A_tmp = self.basis[m](all_types)  # (num_type, 1, num_type)
            A_tmp = torch.transpose(A_tmp, 1, 2)
            if m == 0:
                A_all = A_tmp
            else:
                A_all = torch.cat([A_all, A_tmp], dim=2)
        return A_all

    def forward(self, sample_dict: Dict):
        """
        Calculate
        1) phi_{c_i,c_j}(t_i - t_j) for c_i in "events";
        2) int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
            pHi: (batch_size, num_type) FloatTensor represents sum_{c} sum_{i in history} int_{start}^{stop} phi_cc_i(s)ds
        """
        phi_c = self.intensity(sample_dict)
        pHi = self.expect_counts(sample_dict)
        return phi_c, pHi

    def plot_and_save(self, infect: torch.Tensor, output_name: str = None):
        """
        Plot endogenous impact function for all event types
        Args:
        :param infect: a (num_type, num_type+1, M) FloatTensor containing all endogenous impact
        :param output_name: the name of the output png file
        """
        impact = infect.sum(2).data.numpy()
        plt.figure(figsize=(5, 5))
        plt.imshow(impact)
        plt.colorbar()
        if output_name is None:
            plt.savefig('endogenous_impact.png')
        else:
            plt.savefig(output_name)
        plt.close("all")
        logger.info("Done!")
