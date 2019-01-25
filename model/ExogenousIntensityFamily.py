"""
The classes of 3 typical exogenous intensity functions.

1) constant: mu(t) = mu
2) linear: mu(t) = w^T * feature
3) nonlinear: mu(t) = F(feature), where F can be neural network
....

1-2 can also be nonlinear function when adding a nonlinear activation layer

--- Actually, here users can define their specified exogenous intensity functions. ---

These classes's parent class is "BasicExogenousIntensity".

Written by Hongteng Xu, on Oct. 9, 2018
"""

import torch
import torch.nn as nn
from typing import Dict
from dev.util import logger
from model.ExogenousIntensity import BasicExogenousIntensity
from model.OtherLayers import Identity


class NaiveExogenousIntensity(BasicExogenousIntensity):
    """
    The class of constant exogenous intensity function mu(t) = mu
    """
    def __init__(self, num_type: int, parameter_set: Dict = None):
        """
        Initialize exogenous intensity function: mu(t) = mu, mu in R^{C+1}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        :param parameter_set: a dictionary containing parameters
            parameter_set = {'activation': value = names of activation layers ('identity', 'relu', 'softplus')}
        """
        super(NaiveExogenousIntensity, self).__init__(num_type)
        activation = parameter_set['activation']
        if activation is None:
            self.exogenous_intensity_type = 'constant'
            self.activation = 'identity'
        else:
            self.exogenous_intensity_type = '{}(constant)'.format(activation)
            self.activation = activation

        self.num_type = num_type
        self.dim_embedding = 1
        # self.emb = nn.Embedding(self.num_type, self.dim_embedding, padding_idx=0)
        # self.emb.weight = nn.Parameter(
        #     torch.cat([torch.zeros(1, self.dim_embedding),
        #                torch.FloatTensor(self.num_type - 1, self.dim_embedding).uniform_(0.01 / self.dim_embedding,
        #                                                                                  1 / self.dim_embedding)],
        #               dim=0))
        self.emb = nn.Embedding(self.num_type, self.dim_embedding)
        self.emb.weight = nn.Parameter(
                       torch.FloatTensor(self.num_type, self.dim_embedding).uniform_(0.01 / self.dim_embedding,
                                                                                     1 / self.dim_embedding))
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

    def intensity(self, sample_dict):
        """
        Calculate intensity
        mu_{c_i} for c_i in "events";

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            }
        :return:
            mu_c: (batch_size, 1) FloatTensor represents mu_{c_i};
        """
        events = sample_dict['ci']         # (batch_size, 1)
        mu_c = self.act(self.emb(events))  # (batch_size, 1, 1)
        mu_c = mu_c.squeeze(1)             # (batch_size, 1)
        return mu_c

    def expect_counts(self, sample_dict):
        """
        Calculate the expected number of events in dts
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
        mu_all = self.act(self.emb(all_types))  # (num_type, 1, 1)
        mu_all = mu_all.squeeze(1)  # (num_type, 1)
        mU = torch.matmul(dts, torch.t(mu_all))  # (batch_size, num_type)
        return mU


class LinearExogenousIntensity(BasicExogenousIntensity):
    """
    The class of linear exogenous intensity function mu_c(t) = w_c^T * f.
    Here f is nonnegative feature vector of a sequence.
    """
    def __init__(self, num_type: int, parameter_set: Dict):
        """
        Initialize exogenous intensity function: mu(t) = mu, mu in R^{C+1}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        :param parameter_set: a dictionary containing parameters
            parameter_set = {'activation': value = names of activation layers ('identity', 'relu', 'softplus')
                             'dim_feature': value = the dimension of feature vector (embedding)
                             'num_sequence': the number of sequence}
        """
        super(LinearExogenousIntensity, self).__init__(num_type)
        activation = parameter_set['activation']
        dim_feature = parameter_set['dim_feature']
        num_seq = parameter_set['num_sequence']

        if activation is None:
            self.exogenous_intensity_type = 'w_c^T*f'
            self.activation = 'identity'
        else:
            self.exogenous_intensity_type = '{}(w_c^T*f)'.format(activation)
            self.activation = activation

        self.num_type = num_type
        self.dim_embedding = dim_feature
        self.num_seq = num_seq

        # self.emb = nn.Embedding(self.num_type, self.dim_embedding, padding_idx=0)
        # self.emb.weight = nn.Parameter(
        #     torch.cat([torch.zeros(1, self.dim_embedding),
        #                torch.FloatTensor(self.num_type - 1, self.dim_embedding).uniform_(0.01 / self.dim_embedding,
        #                                                                                  1 / self.dim_embedding)],
        #               dim=0))
        self.emb = nn.Embedding(self.num_type, self.dim_embedding)
        self.emb.weight = nn.Parameter(
                       torch.FloatTensor(self.num_type, self.dim_embedding).uniform_(0.01 / self.dim_embedding,
                                                                                     1 / self.dim_embedding))
        self.emb_seq = nn.Embedding(self.num_seq, self.dim_embedding)
        self.emb_seq.weight = nn.Parameter(
            torch.FloatTensor(self.num_seq, self.dim_embedding).uniform_(0.01 / self.dim_embedding,
                                                                         1 / self.dim_embedding))
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

    def intensity(self, sample_dict):
        """
        Calculate intensity
        mu_{c_i} for c_i in "events";

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            'sn': sequence index (batch_size, 1) LongTensor
            'fsn' features: (batch_size, dim_feature) FloatTensor contains feature vectors of the sequence in the batch
            }
        :return:
            mu_c: (batch_size, 1) FloatTensor represents mu_{c_i};
            mU: (batch_size, num_type) FloatTensor represents int_{0}^{dt} mu_c(s)ds
        """
        events = sample_dict['ci']     # (batch_size, 1)
        features = sample_dict['fsn']  # (batch_size, dim_feature)
        if features is None:
            features = self.emb_seq(sample_dict['sn'])       # (batch_size, 1, dim_feature)
            features = features.squeeze(1)                   # (batch_size, dim_feature)

        mu_c = self.emb(events)        # (batch_size, 1, dim_feature)
        mu_c = mu_c.squeeze(1)         # (batch_size, dim_feature)
        mu_c = mu_c * features         # (batch_size, dim_feature)
        mu_c = mu_c.sum(1)             # (batch_size)
        mu_c = mu_c.view(-1, 1)        # (batch_size, 1)
        mu_c = self.act(mu_c)          # (batch_size, 1)
        return mu_c

    def expect_counts(self, sample_dict):
        """
        Calculate the expected number of events in dts
        int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            'sn': sequence index (batch_size, 1) LongTensor
            'fsn' features: (batch_size, dim_feature) FloatTensor contains feature vectors of the sequence in the batch
            }
        :return:
            mu_c: (batch_size, 1) FloatTensor represents mu_{c_i};
            mU: (batch_size, num_type) FloatTensor represents int_{0}^{dt} mu_c(s)ds
        """
        dts = sample_dict['ti'] - sample_dict['tjs'][:, -1].view(-1, 1)
        all_types = sample_dict['Cs']  # (num_type, 1)
        features = sample_dict['fsn']  # (batch_size, dim_feature)
        if features is None:
            features = self.emb_seq(sample_dict['sn'])       # (batch_size, 1, dim_feature)
            features = features.squeeze(1)                   # (batch_size, dim_feature)

        mu_all = self.emb(all_types)   # (num_type, 1, dim_feature)
        mu_all = mu_all.squeeze(1)     # (num_type, dim_feature)
        mu_all = torch.matmul(features, torch.t(mu_all))  # (batch_size, num_type)
        mu_all = self.act(mu_all)      # (batch_size, num_type)
        mU = mu_all * dts.repeat(1, mu_all.size(1))       # (batch_size, num_type)
        return mU


class NeuralExogenousIntensity(BasicExogenousIntensity):
    """
    The class of neural exogenous intensity function mu_c(t) = F(c, f), where F is a 3-layer neural network,
    c is event type, and f is the feature vector.
    Here, we don't need to ensure f to be nonnegative.
    """
    def __init__(self, num_type: int, parameter_set: Dict):
        """
        Initialize exogenous intensity function: mu(t) = mu, mu in R^{C+1}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        :param parameter_set: a dictionary containing parameters
            parameter_set = {'dim_embedding': the dimension of embeddings.
                             'dim_feature': the dimension of feature vector.
                             'dim_hidden': the dimension of hidden vector.
                             'num_sequence': the number of sequence}
        """
        super(NeuralExogenousIntensity, self).__init__(num_type)
        dim_embedding = parameter_set['dim_embedding']
        dim_feature = parameter_set['dim_feature']
        dim_hidden = parameter_set['dim_hidden']
        num_seq = parameter_set['num_sequence']

        self.exogenous_intensity_type = 'F(c, f)'
        self.num_type = num_type
        self.dim_embedding = dim_embedding
        self.dim_feature = dim_feature
        self.dim_hidden = dim_hidden
        self.num_seq = num_seq

        # self.emb = nn.Embedding(self.num_type, self.dim_embedding, padding_idx=0)
        # self.emb.weight = nn.Parameter(
        #     torch.cat([torch.zeros(1, self.dim_embedding),
        #                torch.FloatTensor(self.num_type - 1, self.dim_embedding).uniform_(0.01 / self.dim_embedding,
        #                                                                                  1 / self.dim_embedding)],
        #               dim=0))
        self.emb = nn.Embedding(self.num_type, self.dim_embedding)
        self.emb.weight = nn.Parameter(
                       torch.FloatTensor(self.num_type, self.dim_embedding).uniform_(0.01 / self.dim_embedding,
                                                                                     1 / self.dim_embedding))
        self.emb_seq = nn.Embedding(self.num_seq, self.dim_embedding)
        self.emb_seq.weight = nn.Parameter(
            torch.cat(torch.FloatTensor(self.num_seq, self.dim_embedding).uniform_(0.01 / self.dim_embedding,
                                                                                   1 / self.dim_embedding)))
        self.softplus = nn.Softplus()
        self.linear1 = nn.Linear(self.dim_embedding, self.dim_hidden)
        self.linear2 = nn.Linear(self.dim_feature, self.dim_hidden)
        self.linear3 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.relu = nn.ReLU()

    def intensity(self, sample_dict):
        """
        Calculate the intensity of event
        mu_{c_i} for c_i in "events";

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            'sn': sequence index (batch_size, 1) LongTensor
            'fsn': features (batch_size, dim_feature) FloatTensor contains feature vectors of the sequence in the batch
            }
        :return:
            mu_c: (batch_size, 1) FloatTensor represents mu_{c_i};
        """
        events = sample_dict['ci']                        # (batch_size, 1)
        features = sample_dict['fsn']                     # (batch_size, dim_feature)
        if features is None:
            features = self.emb_seq(sample_dict['sn'])    # (batch_size, 1, dim_feature)
            features = features.squeeze(1)                # (batch_size, dim_feature)

        event_feat = self.emb(events)                     # (batch_size, 1, dim_embedding)
        event_feat = event_feat.squeeze(1)                # (batch_size, dim_embedding)
        event_feat = self.relu(self.linear1(event_feat))  # (batch_size, dim_hidden)
        seq_feat = self.relu(self.linear2(features))      # (batch_size, dim_hidden)
        seq_feat = self.linear3(seq_feat)                 # (batch_size, dim_hidden)
        feat = seq_feat * event_feat                      # (batch_size, dim_hidden)
        mu_c = self.softplus(feat.sum(1).view(-1, 1))     # (batch_size, 1)
        return mu_c

    def expect_counts(self, sample_dict):
        """
        Calculate the expected number of events in dts
        int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            'sn': sequence index (batch_size, 1) LongTensor
            'fsn': features (batch_size, dim_feature) FloatTensor contains feature vectors of the sequence in the batch
            }
        :return:
            mU: (batch_size, num_type) FloatTensor represents int_{0}^{dt} mu_c(s)ds
        """
        dts = sample_dict['ti'] - sample_dict['tjs'][:, -1].view(-1, 1)
        all_types = sample_dict['Cs']                     # (num_type, 1)
        features = sample_dict['fsn']                     # (batch_size, dim_feature)
        if features is None:
            features = self.emb_seq(sample_dict['sn'])    # (batch_size, 1, dim_feature)
            features = features.squeeze(1)                # (batch_size, dim_feature)

        seq_feat = self.relu(self.linear2(features))      # (batch_size, dim_hidden)
        seq_feat = self.linear3(seq_feat)                 # (batch_size, dim_hidden)
        event_feat_all = self.emb(all_types)              # (num_type, 1, dim_embedding)
        event_feat_all = event_feat_all.squeeze(1)        # (num_type, dim_embedding)
        event_feat_all = self.relu(self.linear1(event_feat_all))  # (num_type, dim_hidden)
        mu_all = torch.matmul(seq_feat, torch.t(event_feat_all))  # (batch_size, num_type)
        mu_all = self.softplus(mu_all)
        mU = mu_all * dts.repeat(1, mu_all.size(1))       # (batch_size, num_type)
        return mU