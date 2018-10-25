"""
Parent class of point process model, which is actually a Poisson process
"""

import copy
import numpy as np
import time
import torch
from dev.util import logger
from preprocess.DataOperation import samples2dict
from model.OtherLayers import MaxLogLike, LeastSquare, CrossEntropy, LowerBoundClipper
import model.ExogenousIntensityFamily


class PointProcessModel(object):
    """
    The class of generalized Hawkes process model
    contains most of necessary function.
    """

    def __init__(self, num_type, mu_dict, loss_type):
        """
        Initialize generalized Hawkes process
        :param num_type: int, the number of event types.
        :param loss_type: str, the type of loss functions
            The length of the list is the number of modalities of the model
            Each element of the list is the number of event categories for each modality
        """
        self.model_name = 'A Poisson Process'
        self.num_type = num_type
        exogenousIntensity = getattr(model.ExogenousIntensityFamily, mu_dict['model_name'])
        self.lambda_model = exogenousIntensity(num_type, mu_dict['parameter_set'])

        self.loss_type = loss_type
        if self.loss_type == 'mle':
            self.loss_function = MaxLogLike()
        elif self.loss_type == 'ls':
            self.loss_function = LeastSquare()
        elif self.loss_type == 'entropy':
            self.loss_function = CrossEntropy()
        else:
            logger.warning('The loss layer is {}, which is not identified... '.format(self.loss_function))
            logger.warning('Maximum likelihood estimation is applied instead.')
            self.loss_function = MaxLogLike()
        # self.print_info()

    def print_info(self):
        """
        Print basic information of the model.
        """
        logger.info(self.model_name)
        self.lambda_model.print_info()
        logger.info("The loss function is {}.".format(self.loss_function))

    def fit(self, dataloader, optimizer, epochs: int, scheduler=None, sparsity: float=None, nonnegative=None,
            use_cuda: bool=False, validation_set=None):
        """
        Learn parameters of a generalized Hawkes process given observed sequences
        :param dataloader: a pytorch batch-based data loader
        :param optimizer: the sgd optimization method defined by PyTorch
        :param epochs: the number of training epochs
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
                loss_total = loss + reg
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

    def validation(self, dataloader, use_cuda):
        """
        Compute the avaraged loss per event of a generalized Hawkes process
        given observed sequences and current model
        :param dataloader: a pytorch batch-based data loader
        :param use_cuda: use cuda (true) or not (false)
        """
        device = torch.device('cuda:0' if use_cuda else 'cpu')
        self.lambda_model.to(device)
        self.lambda_model.eval()

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

        start = time.time()
        loss = 0
        for batch_idx, samples in enumerate(dataloader):
            ci, batch_dict = samples2dict(samples, device, Cs, FCs)
            lambda_t, Lambda_t = self.lambda_model(batch_dict)
            loss += self.loss_function(lambda_t, Lambda_t, ci)

            # display training processes
            if batch_idx % 100 == 0:
                logger.info('Validation [{}/{} ({:.0f}%)]\t Time={:.2f}sec.'.format(
                    batch_idx * ci.size(0), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), time.time() - start))
        return loss/len(dataloader.dataset)

    def simulate(self,
                 history,
                 memory_size: int = 10,
                 time_window: float = 1.0,
                 interval: float = 1.0,
                 max_number: int = 100,
                 use_cuda: bool = False):
        """
        Simulate one or more event sequences from given model.
        :param history: historical observations
            history = {'event_features': None or (C, De) float array of event's static features,
                                  C is the number of event types.
                       'type2idx': a Dict = {'event_name': event_index}
                       'idx2type': a Dict = {event_index: 'event_name'}
                       'seq2idx': a Dict = {'seq_name': seq_index}
                       'idx2seq': a Dict = {seq_index: 'seq_name'}
                       'sequences': a List  = {seq_1, seq_2, ..., seq_N}.
                      }

            For the i-th sequence:
            seq_i = {'times': (N,) float array of timestamps, N is the number of events.
                     'events': (N,) int array of event types.
                     N can be "0" (i.e., no observations)
                     'seq_feature': None or (Ds,) float array of sequence's static feature.
                     't_start': a float number indicating the start timestamp of the sequence.
                     't_stop': a float number indicating the stop timestamp of the sequence.
                     'label': None or int/float number indicating the labels of the sequence}
        :param memory_size: the number of historical events used for simulation
        :param time_window: duration of simulation process.
        :param interval: the interval size calculating the supremum of intensity
        :param max_number: the maximum number of simulated events
        :param use_cuda: use cuda (true) or not (false)
        :return:
            new_data: having the same format as history
            counts: a list of (C,) ndarray, which counts the number of simulated events for each type
        """
        device = torch.device('cuda:0' if use_cuda else 'cpu')
        self.lambda_model.to(device)
        self.lambda_model.eval()

        Cs = torch.LongTensor(list(range(len(history['type2idx']))))
        Cs = Cs.view(-1, 1)
        Cs = Cs.to(device)
        if history['event_features'] is not None:
            all_event_feature = torch.from_numpy(history['event_features'])
            FCs = all_event_feature.type(torch.FloatTensor)
            FCs = torch.t(FCs)  # (num_type, dim_features)
            FCs = FCs.to(device)
        else:
            FCs = None

        t_start = time.time()
        new_data = copy.deepcopy(history)
        # the number of new synthetic events for each type
        counts = np.zeros((self.num_type, len(new_data['sequences'])))
        for i in range(len(new_data['sequences'])):
            times_tmp = []
            events_tmp = []

            # initial point
            new_data['sequences'][i]['t_start'] = history['sequences'][i]['t_stop']
            new_data['sequences'][i]['t_stop'] = history['sequences'][i]['t_stop'] + time_window
            t_now = new_data['sequences'][i]['t_start']+0.01

            # initialize the input of intensity function
            ci = Cs[1:, :]
            ti = torch.FloatTensor([t_now])
            ti = ti.to(device)
            ti = ti.view(1, 1)
            ti = ti.repeat(ci.size(0), 1)

            events = history['sequences'][i]['events']
            times = history['sequences'][i]['times']
            if times is None:
                tjs = torch.FloatTensor([new_data['sequences'][i]['t_start']])
                cjs = torch.LongTensor([0])
            else:
                if memory_size > times.shape[0]:
                    tjs = torch.from_numpy(times)
                    tjs = tjs.type(torch.FloatTensor)
                    cjs = torch.from_numpy(events)
                    cjs = cjs.type(torch.LongTensor)
                else:
                    tjs = torch.from_numpy(times[-memory_size:])
                    tjs = tjs.type(torch.FloatTensor)
                    cjs = torch.from_numpy(events[-memory_size:])
                    cjs = cjs.type(torch.LongTensor)

            tjs = tjs.to(device)
            tjs = tjs.view(1, -1)
            tjs = tjs.repeat(ci.size(0), 1)
            cjs = cjs.to(device)
            cjs = cjs.view(1, -1)
            cjs = cjs.repeat(ci.size(0), 1)

            sn = torch.LongTensor([i])
            sn = sn.to(device)
            sn = sn.view(1, 1)
            sn = sn.repeat(ci.size(0), 1)

            if history['sequences'][i]['seq_feature'] is not None:
                fsn = history['sequences'][i]['seq_feature']
                fsn = torch.from_numpy(fsn)
                fsn = fsn.type(torch.FloatTensor)
                fsn = fsn.view(1, -1).repeat(ci.size(0), 1)
            else:
                fsn = None

            if FCs is None:
                fci = None
                fcjs = None
            else:
                fci = FCs[ci[:, 0], :]
                fcjs = FCs[cjs, :]
                fcjs = torch.transpose(fcjs, 1, 2)

            sample_dict = {'ti': ti,
                           'tjs': tjs,
                           'ci': ci,
                           'cjs': cjs,
                           'sn': sn,
                           'fsn': fsn,
                           'fci': fci,
                           'fcjs': fcjs,
                           'Cs': Cs,
                           'FCs': FCs}

            while t_now < new_data['sequences'][i]['t_stop'] and len(times_tmp) < max_number:
                lambda_t = self.lambda_model.intensity(sample_dict)
                sample_dict['ti'] = sample_dict['ti'] + interval
                lambda_t2 = self.lambda_model.intensity(sample_dict)
                mt = max([float(lambda_t.sum()), float(lambda_t2.sum())])

                s = np.random.exponential(1 / mt)
                if s < interval:
                    sample_dict['ti'] = sample_dict['ti'] + s - interval
                    ti = sample_dict['ti'].numpy()
                    t_now = ti[0, 0]  # float
                    lambda_s = self.lambda_model.intensity(sample_dict)
                    ms = float(lambda_s.sum())

                    u = np.random.rand()
                    ratio = ms/mt
                    if ratio > u:  # generate a new event
                        prob = lambda_s.data.numpy()/ms
                        prob = prob[:, 0]
                        ci = np.random.choice(self.num_type-1, p=prob)+1  # int

                        # add to new sequence
                        times_tmp.append(t_now)
                        events_tmp.append(ci)
                        counts[ci, i] += 1

                        # update batch_dict
                        ti = torch.FloatTensor([t_now])
                        ti = ti.to(device)
                        ti = ti.view(1, 1).repeat(self.num_type-1, 1)
                        ci = torch.LongTensor([ci])
                        ci = ci.to(device)
                        ci = ci.view(1, 1).repeat(self.num_type-1, 1)
                        if memory_size > sample_dict['cjs'].size(1):
                            sample_dict['cjs'] = torch.cat([sample_dict['cjs'], ci], dim=1)
                            sample_dict['tjs'] = torch.cat([sample_dict['tjs'], ti], dim=1)
                        else:
                            sample_dict['cjs'] = torch.cat([sample_dict['cjs'][:, -memory_size+1:], ci], dim=1)
                            sample_dict['tjs'] = torch.cat([sample_dict['tjs'][:, -memory_size+1:], ti], dim=1)
                        if FCs is not None:
                            sample_dict['fcjs'] = FCs[sample_dict['cjs'], :]
                            sample_dict['fcjs'] = torch.transpose(sample_dict['fcjs'], 1, 2)
                else:
                    ti = sample_dict['ti'].numpy()
                    t_now = ti[0, 0]  # float

            if i % 500 == 0:
                logger.info('Sequence {}/{} has been generated... Time={:.2}sec.'.format(
                    i, len(new_data['sequences']), time.time()-t_start))
            times_tmp = np.asarray(times_tmp)
            events_tmp = np.asarray(events_tmp)
            new_data['sequences'][i]['times'] = times_tmp
            new_data['sequences'][i]['events'] = events_tmp
        return new_data, counts

    def predict(self,
                history,
                memory_size: int = 10,
                time_window: float = 1.0,
                interval: float = 1.0,
                max_number: int = 1e5,
                use_cuda: bool = False,
                num_trial: int = 2):
        """
        Predict the expected number of events in the proposed target time window
        :param history: historical observations
            history = {'event_features': None or (C, De) float array of event's static features,
                                  C is the number of event types.
                       'type2idx': a Dict = {'event_name': event_index}
                       'idx2type': a Dict = {event_index: 'event_name'}
                       'seq2idx': a Dict = {'seq_name': seq_index}
                       'idx2seq': a Dict = {seq_index: 'seq_name'}
                       'sequences': a List  = {seq_1, seq_2, ..., seq_N}.
                      }

            For the i-th sequence:
            seq_i = {'times': (N,) float array of timestamps, N is the number of events.
                     'events': (N,) int array of event types.
                     N can be "0" (i.e., no observations)
                     'seq_feature': None or (Ds,) float array of sequence's static feature.
                     'dyn_features': None or (N, Dt) float array of event's dynamic features.
                     't_start': a float number indicating the start timestamp of the sequence.
                     't_stop': a float number indicating the stop timestamp of the sequence.
                     'label': None or int/float number indicating the labels of the sequence}
        :param memory_size: the number of historical events used for simulation
        :param time_window: duration of simulation process.
        :param interval: the interval size calculating the supremum of intensity
        :param max_number: the maximum number of simulated events
        :param use_cuda: whether use cuda or not
        :param num_trial: the number of simulation trials.
        """
        counts_total = np.zeros((self.num_type, len(history['sequences'])))
        for trial in range(num_trial):
            synthetic_data, counts = self.simulate(history,
                                                   memory_size,
                                                   time_window,
                                                   interval,
                                                   max_number,
                                                   use_cuda)
            counts_total += counts

        counts_total /= num_trial
        return counts_total

    def plot_exogenous(self, sample_dict, output_name: str = None):
        """
        Plot and save the stem figure of exogenous intensity
        :param sample_dict:
        :param output_name:
        :return:
        """
        intensity = self.lambda_model.intensity(sample_dict)
        self.lambda_model.plot_and_save(intensity, output_name)

    def save_model(self, full_path, mode: str='entire'):
        """
        Save trained model
        :param full_path: the path of directory
        :param mode: 'parameter' for saving only parameters of the model,
                     'entire' for saving entire model
        """
        if mode == 'entire':
            torch.save(self.lambda_model, full_path)
            logger.info('The entire model is saved in {}.'.format(full_path))
        elif mode == 'parameter':
            torch.save(self.lambda_model.state_dict(), full_path)
            logger.info('The parameters of the model is saved in {}.'.format(full_path))
        else:
            logger.warning("'{}' is a undefined mode, we use 'entire' mode instead.".format(mode))
            torch.save(self.lambda_model, full_path)
            logger.info('The entire model is saved in {}.'.format(full_path))

    def load_model(self, full_path, mode: str='entire'):
        """
        Load pre-trained model
        :param full_path: the path of directory
        :param mode: 'parameter' for saving only parameters of the model,
                     'entire' for saving entire model
        """
        if mode == 'entire':
            self.lambda_model = torch.load(full_path)
        elif mode == 'parameter':
            self.lambda_model.load_state_dict(torch.load(full_path))
        else:
            logger.warning("'{}' is a undefined mode, we use 'entire' mode instead.".format(mode))
            self.lambda_model = torch.load(full_path)
