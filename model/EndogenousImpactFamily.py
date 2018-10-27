"""
The classes of 5 typical endogenous impact functions.

1) basis representation: phi_{cc'}(t) = sum_m a_{cc'm} * kernel_m(t)
2) factorization: phi_{cc'}(t) = sum_m (u_{cm}^T * v_{c'm}) * kernel_m(t)
3) linear: phi_{cc'}(t) = sum_m (w_{cm}^T * f_{c'}) * kernel_m(t)
4) bilinear: phi_{cc'}(t) = sum_m (f_{c}^T * W_m * f_{c'}) * kernel_m(t)
5) neural: mu(t) = F(feature) * kernel, where F can be neural network

1-4 can also be nonlinear model when adding nonlinear activation layer
....
--- Actually, here users can define their specified endogenous impact functions. ---

These classes's parent class is "BasicExogenousIntensity".

Written by Hongteng Xu, on Oct. 9, 2018
"""

from dev.util import logger
import torch
import torch.nn as nn
from typing import Dict
from model.EndogenousImpact import BasicEndogenousImpact
from model.OtherLayers import Identity


class NaiveEndogenousImpact(BasicEndogenousImpact):
    """
    The class of naive endogenous impact functions sum_i phi_{kk_i}(t-t_i) for k = 1,...,C,
    which actually a simple endogenous impact with phi_{kk'}(t) = sum_{m} a_{kk'm} kernel_m(t)
    """

    def __init__(self, num_type: int, kernel, parameter_set: Dict):
        """
        Initialize endogenous impact: phi_{kk'}(t) = sum_{m} a_{kk'm} kernel_m(t),
        for m = 1, ..., M, A_m = [a_{kk'm}] in R^{C*C+1}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        :param kernel: an instance of a decay kernel class in "DecayKernelFamily"
        :param parameter_set: a dictionary containing parameters
            parameter_set = {'activation': value = names of activation layers ('identity', 'relu', 'softplus')}
        """
        super(NaiveEndogenousImpact, self).__init__(num_type, kernel)
        activation = parameter_set['activation']
        if activation is None:
            self.endogenous_impact_type = "sum_m a_(kk'm) * kernel_m(t)"
            self.activation = 'identity'
        else:
            self.endogenous_impact_type = "sum_m {}(a_(kk'm)) * kernel_m(t))".format(activation)
            self.activation = activation

        self.decay_kernel = kernel
        self.num_base = self.decay_kernel.parameters.shape[1]
        self.num_type = num_type
        self.dim_embedding = num_type
        for m in range(self.num_base):
            emb = nn.Embedding(self.num_type, self.dim_embedding, padding_idx=0)
            emb.weight = nn.Parameter(
                torch.cat([torch.zeros(self.num_type, 1),
                           torch.FloatTensor(self.num_type, self.dim_embedding - 1).uniform_(0.01 / self.dim_embedding,
                                                                                             1 / self.dim_embedding)],
                          dim=1))
            if m == 0:
                self.basis = nn.ModuleList([emb])
            else:
                self.basis.append(emb)

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

    def intensity(self, sample_dict: Dict):
        """
        Calculate the intensity of events
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
            pHi: (batch_size, num_type) FloatTensor represents sum_{c, i in history} int_{start}^{stop} phi_cc_i(s)ds
        """
        event_time = sample_dict['ti']     # (batch_size, 1)
        history_time = sample_dict['tjs']  # (batch_size, memory_size)
        events = sample_dict['ci']         # (batch_size, 1)
        history = sample_dict['cjs']       # (batch_size, memory_size)

        dts = event_time.repeat(1, history_time.size(1)) - history_time  # (batch_size, memory_size)
        # gt = self.decay_kernel.values(dts.numpy())
        # gt = torch.from_numpy(gt)
        # gt = gt.type(torch.FloatTensor)                                  # (batch_size, memory_size, num_base)
        gt = self.decay_kernel.values(dts)

        phi_c = 0
        for m in range(self.num_base):
            A_cm = self.basis[m](events)                        # (batch_size, 1, dim_embedding)
            A_cm = A_cm.squeeze(1)                              # (batch_size, dim_embedding)
            A_cm = A_cm.gather(1, history)                      # (batch_size, memory_size)
            A_cm = self.act(A_cm)
            A_cm = A_cm.unsqueeze(1)                            # (batch_size, 1, memory_size)
            phi_c += torch.bmm(A_cm, gt[:, :, m].unsqueeze(2))  # (batch_size, 1, 1)
        phi_c = phi_c[:, :, 0]
        return phi_c

    def expect_counts(self, sample_dict: Dict):
        """
        Calculate the expected number of events in dts
        int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
            pHi: (batch_size, num_type) FloatTensor represents sum_{c, i in history} int_{start}^{stop} phi_cc_i(s)ds
        """
        event_time = sample_dict['ti']     # (batch_size, 1)
        history_time = sample_dict['tjs']  # (batch_size, memory_size)
        history = sample_dict['cjs']       # (batch_size, memory_size)
        all_types = sample_dict['Cs']      # (num_type, 1)

        dts = event_time.repeat(1, history_time.size(1)) - history_time     # (batch_size, memory_size)
        last_time = history_time[:, -1].unsqueeze(1)
        t_start = last_time.repeat(1, history_time.size(1)) - history_time  # (batch_size, memory_size)
        t_stop = dts                                                        # (batch_size, memory_size)
        # Gt = self.decay_kernel.integrations(t_stop.numpy(), t_start.numpy())
        # Gt = torch.from_numpy(Gt)
        # Gt = Gt.type(torch.FloatTensor)                                     # (batch_size, memory_size, num_base)
        Gt = self.decay_kernel.integrations(t_stop, t_start)

        pHi = 0
        history2 = history.unsqueeze(1).repeat(1, all_types.size(0), 1)     # (batch_size, num_type, memory_size)
        for m in range(self.num_base):
            A_all = self.basis[m](all_types)                    # (num_type, 1, dim_embedding)
            A_all = A_all.squeeze(1).unsqueeze(0)               # (1, num_type, dim_embedding)
            A_all = A_all.repeat(Gt.size(0), 1, 1)              # (batch_size, num_type, dim_embedding)
            A_all = A_all.gather(2, history2)                   # (batch_size, num_type, memory_size)
            A_all = self.act(A_all)
            pHi += torch.bmm(A_all, Gt[:, :, m].unsqueeze(2))   # (batch_size, num_type, 1)
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
            A_tmp = self.act(torch.transpose(A_tmp, 1, 2))
            if m == 0:
                A_all = A_tmp
            else:
                A_all = torch.cat([A_all, A_tmp], dim=2)
        return A_all


class FactorizedEndogenousImpact(BasicEndogenousImpact):
    """
    The class of factorized endogenous impact functions
    phi_{cc'}(t) = sum_m (u_{cm}^T * v_{c'm}) * kernel_m(t)
    Here, U_m=[u_{cm}] and V_m=[v_{cm}], m=1,...,M, are embedding matrices
    """

    def __init__(self, num_type: int, kernel, parameter_set: Dict):
        """
        Initialize endogenous impact: phi_{kk'}(t) = sum_{m} a_{kk'm} kernel_m(t),
        for m = 1, ..., M, A_m = [a_{kk'm}] in R^{C*C+1}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        :param kernel: an instance of a decay kernel class in "DecayKernelFamily"
        :param parameter_set: a dictionary containing parameters
            parameter_set = {'activation': value = names of activation layers ('identity', 'relu', 'softplus')
                             'dim_feature': value = the dimension of feature vector (embedding)}
        """
        super(FactorizedEndogenousImpact, self).__init__(num_type, kernel)
        activation = parameter_set['activation']
        dim_embedding = parameter_set['dim_embedding']
        if activation is None:
            self.endogenous_impact_type = "sum_m (u_{cm}^T * v_{c'm}) * kernel_m(t)"
            self.activation = 'identity'
        else:
            self.endogenous_impact_type = "sum_m {}(u_(cm)^T * v_(c'm)) * kernel_m(t))".format(activation)
            self.activation = activation

        self.decay_kernel = kernel
        self.num_base = self.decay_kernel.parameters.shape[1]
        self.num_type_u = num_type
        self.num_type_v = num_type
        self.dim_embedding = dim_embedding
        for m in range(self.num_base):
            emb_u = nn.Embedding(self.num_type_u, self.dim_embedding, padding_idx=0)
            emb_v = nn.Embedding(self.num_type_v, self.dim_embedding, padding_idx=0)
            emb_u.weight = nn.Parameter(
                torch.cat([torch.zeros(1, self.dim_embedding),
                           torch.FloatTensor(self.num_type_u-1, self.dim_embedding).uniform_(
                               0.01 / self.dim_embedding,
                               1 / self.dim_embedding)],
                          dim=0))
            emb_v.weight = nn.Parameter(
                torch.cat([torch.zeros(1, self.dim_embedding),
                           torch.FloatTensor(self.num_type_v-1, self.dim_embedding).uniform_(
                               0.01 / self.dim_embedding,
                               1 / self.dim_embedding)],
                          dim=0))
            if m == 0:
                self.basis_u = nn.ModuleList([emb_u])
                self.basis_v = nn.ModuleList([emb_v])
            else:
                self.basis_u.append(emb_u)
                self.basis_v.append(emb_v)

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

    def intensity(self, sample_dict: Dict):
        """
        Calculate the intensity of event
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
        """
        event_time = sample_dict['ti']     # (batch_size, 1)
        history_time = sample_dict['tjs']  # (batch_size, memory_size)
        events = sample_dict['ci']         # (batch_size, 1)
        history = sample_dict['cjs']       # (batch_size, memory_size)

        dts = event_time.repeat(1, history_time.size(1)) - history_time  # (batch_size, memory_size)
        gt = self.decay_kernel.values(dts)
        # gt = torch.from_numpy(gt)
        # gt = gt.type(torch.FloatTensor)                                  # (batch_size, memory_size, num_base)

        phi_c = 0
        for m in range(self.num_base):
            u_cm = self.basis_u[m](events)           # (batch_size, 1, dim_embedding)
            v_cm = self.basis_v[m](history)          # (batch_size, memory_size, dim_embedding)
            v_cm = torch.transpose(v_cm, 1, 2)       # (batch_size, dim_embedding, memory_size)
            A_cm = torch.bmm(u_cm, v_cm)             # (batch_size, 1, memory_size)
            A_cm = self.act(A_cm)
            phi_c += torch.bmm(A_cm, gt[:, :, m].unsqueeze(2))  # (batch_size, 1, 1)
        phi_c = phi_c[:, :, 0]
        return phi_c

    def expect_counts(self, sample_dict: Dict):
        """
        Calculate the expected number of events in dts
        int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            }
        :return:
            pHi: (batch_size, num_type) FloatTensor represents sum_{c, i in history} int_{start}^{stop} phi_cc_i(s)ds
        """
        event_time = sample_dict['ti']     # (batch_size, 1)
        history_time = sample_dict['tjs']  # (batch_size, memory_size)
        history = sample_dict['cjs']       # (batch_size, memory_size)
        all_types = sample_dict['Cs']      # (num_type, 1)

        dts = event_time.repeat(1, history_time.size(1)) - history_time  # (batch_size, memory_size)
        last_time = history_time[:, -1].unsqueeze(1)
        t_start = last_time.repeat(1, history_time.size(1)) - history_time  # (batch_size, memory_size)
        t_stop = dts                                                     # (batch_size, memory_size)
        Gt = self.decay_kernel.integrations(t_stop, t_start)
        # Gt = torch.from_numpy(Gt)
        # Gt = Gt.type(torch.FloatTensor)                                  # (batch_size, memory_size, num_base)

        pHi = 0
        for m in range(self.num_base):
            v_cm = self.basis_v[m](history)          # (batch_size, memory_size, dim_embedding)
            v_cm = torch.transpose(v_cm, 1, 2)       # (batch_size, dim_embedding, memory_size)
            u_all = self.basis_u[m](all_types)       # (num_type, 1, dim_embedding)
            u_all = torch.transpose(u_all, 0, 1)     # (1, num_type, dim_embedding)
            u_all = u_all.repeat(Gt.size(0), 1, 1)   # (batch_size, num_type, dim_embedding)
            A_all = torch.matmul(u_all, v_cm)        # (batch_size, num_type, memory_size)
            A_all = self.act(A_all)
            pHi += torch.bmm(A_all, Gt[:, :, m].unsqueeze(2))   # (batch_size, num_type, 1)
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
        A_all = 0
        all_types = sample_dict['Cs'][:, 0]  # (num_type, )
        for m in range(self.num_base):
            u_all = self.basis_u[m](all_types)  # (num_type, dim_embedding)
            v_all = self.basis_v[m](all_types)  # (num_type, dim_embedding)
            A_tmp = torch.matmul(u_all, torch.t(v_all)).unsqueeze(2)  # (num_type, num_type, 1)
            A_tmp = self.act(A_tmp)
            if m == 0:
                A_all = A_tmp
            else:
                A_all = torch.cat([A_all, A_tmp], dim=2)
        return A_all


class LinearEndogenousImpact(BasicEndogenousImpact):
    """
    The class of linear endogenous impact functions
    phi_{cc'}(t) = sum_m (w_{cm}^T * f_{c'}) * kernel_m(t)
    Here W_m = [w_{cm}], for m=1,...,M, are embedding matrices
    f_{c'} is the feature vector associated with the c'-th history event
    """

    def __init__(self, num_type: int, kernel, parameter_set: Dict):
        """
        Initialize endogenous impact: phi_{kk'}(t) = sum_m (w_{cm}^T * f_{c'}) * kernel_m(t),
        for m = 1, ..., M, W_m = [w_{cm}] in R^{(C+1)*D}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        :param kernel: an instance of a decay kernel class in "DecayKernelFamily"
        :param parameter_set: a dictionary containing parameters
            parameter_set = {'activation': value = names of activation layers ('identity', 'relu', 'softplus')
                             'dim_feature': value = the dimension of feature vector (embedding)}
        """
        super(LinearEndogenousImpact, self).__init__(num_type, kernel)
        activation = parameter_set['activation']
        dim_feature = parameter_set['dim_feature']
        if activation is None:
            self.endogenous_impact_type = "sum_m (u_{cm}^T * v_{c'm}) * kernel_m(t)"
            self.activation = 'identity'
        else:
            self.endogenous_impact_type = "sum_m {}(u_(cm)^T * v_(c'm)) * kernel_m(t))".format(activation)
            self.activation = activation

        self.decay_kernel = kernel
        self.num_base = self.decay_kernel.parameters.shape[1]
        self.num_type = num_type
        self.dim_embedding = dim_feature
        for m in range(self.num_base):
            emb = nn.Embedding(self.num_type, self.dim_embedding, padding_idx=0)
            emb.weight = nn.Parameter(
                torch.cat([torch.zeros(1, self.dim_embedding),
                           torch.FloatTensor(self.num_type-1, self.dim_embedding).uniform_(0.01 / self.dim_embedding,
                                                                                           1 / self.dim_embedding)],
                          dim=0))
            if m == 0:
                self.basis = nn.ModuleList([emb])
            else:
                self.basis.append(emb)

        self.emb_event = nn.Embedding(self.num_type, self.dim_embedding, padding_idx=0)
        self.emb_event.weight = nn.Parameter(
            torch.cat([torch.zeros(1, self.dim_embedding),
                       torch.FloatTensor(self.num_type-1, self.dim_embedding).uniform_(0.01 / self.dim_embedding,
                                                                                       1 / self.dim_embedding)],
                      dim=0))

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

    def intensity(self, sample_dict: Dict):
        """
        Calculate the intensity of events
        phi_{c_i,c_j}(t_i - t_j) for c_i in "events";

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'fcjs': history_features (batch_size, Dc, memory_size) FloatTensor of historical features
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
        """
        event_time = sample_dict['ti']      # (batch_size, 1)
        history_time = sample_dict['tjs']   # (batch_size, memory_size)
        events = sample_dict['ci']          # (batch_size, 1)
        history = sample_dict['cjs']        # (batch_size, memory_size)
        history_feat = sample_dict['fcjs']  # (batch_size, dim_feature, memory_size)
        if history_feat is None:
            history_feat = self.emb_event(history)  # (batch_size, memory_size, dim_feature)
            history_feat = torch.transpose(history_feat, 1, 2)  # (batch_size, dim_feature, memory_size)

        dts = event_time.repeat(1, history_time.size(1)) - history_time  # (batch_size, memory_size)
        gt = self.decay_kernel.values(dts)
        # gt = torch.from_numpy(gt)
        # gt = gt.type(torch.FloatTensor)                                  # (batch_size, memory_size, num_base)

        phi_c = 0
        for m in range(self.num_base):
            u_cm = self.basis[m](events)                 # (batch_size, 1, dim_feature)
            # print(u_cm.size())
            # print(history_feat.size())
            A_cm = torch.bmm(u_cm, history_feat)         # (batch_size, 1, memory_size)
            A_cm = self.act(A_cm)
            phi_c += torch.bmm(A_cm, gt[:, :, m].unsqueeze(2))  # (batchsize, 1, 1)
        phi_c = phi_c[:, :, 0]
        return phi_c

    def expect_counts(self, sample_dict: Dict):
        """
        Calculate the expected number of events in dts
        int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            'fcjs': history_features (batch_size, Dc, memory_size) FloatTensor of historical features
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
            pHi: (batch_size, num_type) FloatTensor represents sum_{c, i in history} int_{start}^{stop} phi_cc_i(s)ds
        """
        event_time = sample_dict['ti']      # (batch_size, 1)
        history_time = sample_dict['tjs']   # (batch_size, memory_size)
        history = sample_dict['cjs']        # (batch_size, memory_size)
        all_types = sample_dict['Cs']       # (num_type, 1)
        history_feat = sample_dict['fcjs']  # (batch_size, dim_feature, memory_size)
        if history_feat is None:
            history_feat = self.emb_event(history)  # (batch_size, memory_size, dim_feature)
            history_feat = torch.transpose(history_feat, 1, 2)  # (batch_size, dim_feature, memory_size)

        dts = event_time.repeat(1, history_time.size(1)) - history_time  # (batch_size, memory_size)
        last_time = history_time[:, -1].unsqueeze(1)
        t_start = last_time.repeat(1, history_time.size(1)) - history_time  # (batch_size, memory_size)
        t_stop = dts  # (batch_size, memory_size)
        Gt = self.decay_kernel.integrations(t_stop, t_start)
        # Gt = torch.from_numpy(Gt)
        # Gt = Gt.type(torch.FloatTensor)                                  # (batch_size, memory_size, num_base)

        pHi = 0
        for m in range(self.num_base):
            u_all = self.basis[m](all_types)             # (num_type, 1, dim_embedding)
            u_all = u_all.squeeze(1)                     # (num_type, dim_embedding)
            # print(u_all.size())
            # print(history_feat.size())
            A_all = torch.matmul(u_all, history_feat)    # (batchsize, num_type, memory_size)
            A_all = self.act(A_all)
            pHi += torch.bmm(A_all, Gt[:, :, m].unsqueeze(2))   # (batchsize, num_type, 1)
        pHi = pHi[:, :, 0]
        return pHi

    def granger_causality(self, sample_dict: Dict):
        """
        Calculate the granger causality among event types
        a_{cc'm}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            'FCs': all_features (num_type, dim_feature)
            }
        :return:
            A_all: (num_type, num_type, num_base) FloatTensor represents a_{cc'm} in phi_{cc'}(t)
        """
        A_all = 0
        all_types = sample_dict['Cs'][:, 0]  # (num_type, )
        all_features = sample_dict['FCs']  # (num_type, dim_feature)
        if all_features is None:
            all_features = self.emb_event(all_types)  # (num_type, dim_feature)

        for m in range(self.num_base):
            u_all = self.basis[m](all_types)  # (num_type, dim_feature)
            A_tmp = torch.matmul(u_all, torch.t(all_features)).unsqueeze(2)  # (num_type, num_type, 1)
            A_tmp = self.act(A_tmp)
            if m == 0:
                A_all = A_tmp
            else:
                A_all = torch.cat([A_all, A_tmp], dim=2)
        return A_all


class BilinearEndogenousImpact(BasicEndogenousImpact):
    """
    The class of bilinear endogenous impact functions
    phi_{cc'}(t) = sum_m (f_{c}^T * W_m * f_{c'}) * kernel_m(t)
    Here W_m for m=1,...,M, are embedding matrices
    f_{c'} is the feature vector associated with the c'-th history event
    """

    def __init__(self, num_type: int, kernel, parameter_set: Dict):
        """
        Initialize endogenous impact: phi_{cc'}(t) = sum_m (f_{c}^T * W_m * f_{c'}) * kernel_m(t)
        for m = 1, ..., M, W_m = [w_{cm}] in R^{(C+1)*D}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        :param kernel: an instance of a decay kernel class in "DecayKernelFamily"
        :param parameter_set: a dictionary containing parameters
            parameter_set = {'activation': value = names of activation layers ('identity', 'relu', 'softplus')
                             'dim_feature': value = the dimension of feature vector (embedding)}
        """
        super(BilinearEndogenousImpact, self).__init__(num_type, kernel)
        activation = parameter_set['activation']
        dim_feature = parameter_set['dim_feature']
        if activation is None:
            self.endogenous_impact_type = "sum_m (f_{c}^T * W_m * f_{c'}) * kernel_m(t)"
            self.activation = 'identity'
        else:
            self.endogenous_impact_type = "sum_m {}(f_(c)^T * W_m * f_(c')) * kernel_m(t))".format(activation)
            self.activation = activation

        self.decay_kernel = kernel
        self.num_base = self.decay_kernel.parameters.shape[1]
        self.num_type = num_type
        self.dim_embedding = dim_feature
        for m in range(self.num_base):
            emb = nn.Linear(self.dim_embedding, self.dim_embedding, bias=False)
            emb.weight = nn.Parameter(
                torch.FloatTensor(self.dim_embedding, self.dim_embedding).uniform_(0.01 / self.dim_embedding,
                                                                                   1 / self.dim_embedding))
            if m == 0:
                self.basis = nn.ModuleList([emb])
            else:
                self.basis.append(emb)

        self.emb_event = nn.Embedding(self.num_type, self.dim_embedding, padding_idx=0)
        self.emb_event.weight = nn.Parameter(
            torch.cat([torch.zeros(1, self.dim_embedding),
                       torch.FloatTensor(self.num_type - 1, self.dim_embedding).uniform_(0.01 / self.dim_embedding,
                                                                                         1 / self.dim_embedding)],
                      dim=0))

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

    def intensity(self, sample_dict: Dict):
        """
        Calculate the intensity of event
        phi_{c_i,c_j}(t_i - t_j) for c_i in "events";

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'fci': current_feature (batch_size, Dc) FloatTensor of current feature
            'fcjs': history_features (batch_size, Dc, memory_size) FloatTensor of historical features
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
        """

        event_time = sample_dict['ti']      # (batch_size, 1)
        history_time = sample_dict['tjs']   # (batch_size, memory_size)
        events = sample_dict['ci']          # (batch_size, 1)
        history = sample_dict['cjs']        # (batch_size, memory_size)
        current_feat = sample_dict['fci']   # (batch_size, dim_feature)
        history_feat = sample_dict['fcjs']  # (batch_size, dim_feature, memory_size)
        if history_feat is None:
            current_feat = self.emb_event(events)
            current_feat = current_feat.squeeze(1)  # (batch_size, dim_feature)
            history_feat = self.emb_event(history)  # (batch_size, memory_size, dim_feature)
            history_feat = torch.transpose(history_feat, 1, 2)  # (batch_size, dim_feature, memory_size)

        dts = event_time.repeat(1, history_time.size(1)) - history_time  # (batch_size, memory_size)
        gt = self.decay_kernel.values(dts)
        # gt = torch.from_numpy(gt)
        # gt = gt.type(torch.FloatTensor)                                  # (batch_size, memory_size, num_base)

        phi_c = 0
        for m in range(self.num_base):
            u_cm = self.basis[m](current_feat)           # (batchsize, dim_feature)
            u_cm = u_cm.unsqueeze(1)                     # (batchsize, 1, dim_embedding)
            A_cm = torch.bmm(u_cm, history_feat)         # (batchsize, 1, memory_size)
            A_cm = self.act(A_cm)
            phi_c += torch.bmm(A_cm, gt[:, :, m].unsqueeze(2))    # (batchsize, 1, 1)
        phi_c = phi_c[:, :, 0]
        return phi_c

    def expect_counts(self, sample_dict: Dict):
        """
        Calculate the expected number of events in dts
        int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            'fcjs': history_features (batch_size, Dc, memory_size) FloatTensor of historical features
            'FCs': all_feats (num_type, dim_feature) FloatTensor of all event types
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
            pHi: (batch_size, num_type) FloatTensor represents sum_{c, i in history} int_{start}^{stop} phi_cc_i(s)ds
        """

        event_time = sample_dict['ti']      # (batch_size, 1)
        history_time = sample_dict['tjs']   # (batch_size, memory_size)
        history = sample_dict['cjs']        # (batch_size, memory_size)
        all_types = sample_dict['Cs']       # (num_type, 1)
        all_feats = sample_dict['FCs']      # (num_type, dim_feature)
        history_feat = sample_dict['fcjs']  # (batch_size, dim_feature, memory_size)
        if history_feat is None:
            all_feats = self.emb_event(all_types)
            all_feats = all_feats.squeeze(1)        # (num_type, dim_feature)
            history_feat = self.emb_event(history)  # (batch_size, memory_size, dim_feature)
            history_feat = torch.transpose(history_feat, 1, 2)  # (batch_size, dim_feature, memory_size)

        dts = event_time.repeat(1, history_time.size(1)) - history_time  # (batch_size, memory_size)
        last_time = history_time[:, -1].unsqueeze(1)
        t_start = last_time.repeat(1, history_time.size(1)) - history_time  # (batch_size, memory_size)
        t_stop = dts                                                                  # (batch_size, memory_size)
        Gt = self.decay_kernel.integrations(t_stop, t_start)
        # Gt = torch.from_numpy(Gt)
        # Gt = Gt.type(torch.FloatTensor)                                  # (batch_size, memory_size, num_base)

        pHi = 0
        for m in range(self.num_base):
            u_all = self.basis[m](all_feats)             # (num_type, dim_feature)
            A_all = torch.matmul(u_all, history_feat)    # (batchsize, num_type, memory_size)
            A_all = self.act(A_all)
            pHi += torch.bmm(A_all, Gt[:, :, m].unsqueeze(2))     # (batchsize, num_type, 1)
        pHi = pHi[:, :, 0]
        return pHi

    def granger_causality(self, sample_dict: Dict):
        """
        Calculate the granger causality among event types
        a_{cc'm}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            'FCs': all_features (num_type, dim_feature)
            }
        :return:
            A_all: (num_type, num_type, num_base) FloatTensor represents a_{cc'm} in phi_{cc'}(t)
        """
        A_all = 0
        all_types = sample_dict['Cs'][:, 0]  # (num_type, )
        all_features = sample_dict['FCs']  # (num_type, dim_feature)
        if all_features is None:
            all_features = self.emb_event(all_types)  # (num_type, dim_feature)

        for m in range(self.num_base):
            u_all = self.basis[m](all_features)  # (num_type, dim_feature)
            A_tmp = torch.matmul(u_all, torch.t(all_features)).unsqueeze(2)  # (num_type, num_type, 1)
            A_tmp = self.act(A_tmp)
            if m == 0:
                A_all = A_tmp
            else:
                A_all = torch.cat([A_all, A_tmp], dim=2)
        return A_all