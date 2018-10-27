"""
An example of a generalized Hawkes process model with features
"""

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import dev.util as util
from model.HawkesProcess import HawkesProcessModel
from preprocess.DataIO import load_sequences_csv, load_event_features_csv, load_seq_features_csv
from preprocess.DataOperation import data_info, EventSampler, enumerate_all_events

if __name__ == '__main__':
    # hyper-parameters
    memory_size = 3
    batch_size = 128
    use_cuda = True
    use_cuda = use_cuda and torch.cuda.is_available()
    seed = 1
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    epochs = 3

    # test sequence loading functions
    # load event sequences from csv file
    domain_names = {'seq_id': 'id',
                    'time': 'time',
                    'event': 'event'}
    database = load_sequences_csv('{}/{}/Linkedin.csv'.format(util.POPPY_PATH, util.DATA_DIR),
                                  domain_names=domain_names)
    data_info(database)

    # load sequences' features
    domain_dict = {'time': 'numerical',
                   'option1': 'categorical'}
    database = load_seq_features_csv('{}/{}/Linkedin.csv'.format(util.POPPY_PATH, util.DATA_DIR),
                                     seq_domain='id',
                                     domain_dict=domain_dict,
                                     database=database,
                                     normalize=1)
    data_info(database)

    # load event types' features
    database = load_event_features_csv('{}/{}/Linkedin.csv'.format(util.POPPY_PATH, util.DATA_DIR),
                                       event_domain='event',
                                       domain_dict=domain_dict,
                                       database=database,
                                       normalize=1)
    data_info(database)

    # sample batches from database
    trainloader = DataLoader(EventSampler(database=database, memorysize=memory_size),
                             batch_size=batch_size,
                             shuffle=True,
                             **kwargs)
    validloader = DataLoader(EventSampler(database=database, memorysize=memory_size),
                             batch_size=batch_size,
                             shuffle=True,
                             **kwargs)

    # initialize model
    num_type = len(database['type2idx'])
    dim_feature_mu = database['sequences'][0]['seq_feature'].shape[0]
    dim_feature_alpha = database['event_features'].shape[0]
    mu_dict = {'model_name': 'LinearExogenousIntensity',
               'parameter_set': {'activation': 'identity',
                                 'dim_feature': dim_feature_mu,
                                 'num_sequence': len(database['seq2idx'])}
               }
    alpha_dict = {'model_name': 'BilinearEndogenousImpact',
                  'parameter_set': {'activation': 'identity',
                                    'dim_feature': dim_feature_alpha}
                  }
    kernel_para = np.zeros((2, 1))
    kernel_para[1, 0] = 0.5
    kernel_para = torch.from_numpy(kernel_para)
    kernel_para = kernel_para.type(torch.FloatTensor)
    kernel_dict = {'model_name': 'ExponentialKernel',
                   'parameter_set': kernel_para}
    loss_type = 'mle'
    hawkes_model = HawkesProcessModel(num_type=num_type,
                                      mu_dict=mu_dict,
                                      alpha_dict=alpha_dict,
                                      kernel_dict=kernel_dict,
                                      activation='identity',
                                      loss_type=loss_type,
                                      use_cuda=use_cuda)

    # initialize optimizer
    optimizer = optim.Adam(hawkes_model.lambda_model.parameters(), lr=0.001)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # train model
    hawkes_model.fit(trainloader, optimizer, epochs, scheduler=scheduler,
                     sparsity=10, nonnegative=0, use_cuda=use_cuda, validation_set=validloader)
    # save model
    hawkes_model.save_model('{}/{}/full.pt'.format(util.POPPY_PATH, util.OUTPUT_DIR), mode='entire')
    hawkes_model.save_model('{}/{}/para.pt'.format(util.POPPY_PATH, util.OUTPUT_DIR), mode='parameter')

    # load model
    hawkes_model.load_model('{}/{}/full.pt'.format(util.POPPY_PATH, util.OUTPUT_DIR), mode='entire')

    # plot exogenous intensity
    all_events = enumerate_all_events(database, seq_id=1, use_cuda=use_cuda)
    hawkes_model.plot_exogenous(all_events,
                                output_name='{}/{}/exogenous_featureHawkes.png'.format(util.POPPY_PATH, util.OUTPUT_DIR))

    # plot endogenous Granger causality
    hawkes_model.plot_causality(all_events,
                                output_name='{}/{}/causality_featureHawkes.png'.format(util.POPPY_PATH, util.OUTPUT_DIR))

    # simulate new data based on trained model
    new_data, counts = hawkes_model.simulate(history=database,
                                             memory_size=memory_size,
                                             time_window=5,
                                             interval=1.0,
                                             max_number=10,
                                             use_cuda=use_cuda)