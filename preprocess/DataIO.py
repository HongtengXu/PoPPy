"""
This script contains the function loading data from csv file
"""

from dev.util import logger
import numpy as np
import pandas as pd
import time
from typing import Dict


def load_sequences_csv(file_name: str, domain_names: Dict):
    """
    Load event sequences from a csv file
    :param file_name: the path and name of the target csv file
    :param domain_names: a dictionary contains the names of the key columns
                         corresponding to {'seq_id', 'time', 'event'}
        The format should be
        domain_names = {'seq_id': the column name of sequence name,
                        'time': the column name of timestamps,
                        'event': the column name of events}
    :return: database: a dictionary containing observed event sequences
        database = {'event_features': None,
                    'type2idx': a Dict = {'event_name': event_index}
                    'idx2type': a Dict = {event_index: 'event_name'}
                    'seq2idx': a Dict = {'seq_name': seq_index}
                    'idx2seq': a Dict = {seq_index: 'seq_name'}
                    'sequences': a List  = [seq_1, seq_2, ..., seq_N].
                    }

        For the i-th sequence:
        seq_i = {'times': (N,) float array of timestamps, N is the number of events.
                 'events': (N,) int array of event types.
                 'seq_feature': None.
                 't_start': a float number, the start timestamp of the sequence.
                 't_stop': a float number, the stop timestamp of the sequence.
                 'label': None
                 }
    """
    database = {'event_features': None,
                'type2idx': None,
                'idx2type': None,
                'seq2idx': None,
                'idx2seq': None,
                'sequences': []}

    df = pd.read_csv(file_name)
    type2idx = {}
    idx2type = {}
    seq2idx = {}
    idx2seq = {}

    logger.info('Count the number of sequences...')
    start = time.time()
    seq_idx = 0
    type_idx = 0
    for i, row in df.iterrows():
        seq_name = str(row[domain_names['seq_id']])
        event_type = str(row[domain_names['event']])
        if seq_name not in seq2idx.keys():
            seq2idx[seq_name] = seq_idx
            seq = {'times': [],
                   'events': [],
                   'seq_feature': None,
                   't_start': 0.0,
                   't_stop': 0.0,
                   'label': None}
            database['sequences'].append(seq)
            seq_idx += 1

        if event_type not in type2idx.keys():
            type2idx[event_type] = type_idx
            type_idx += 1

        if i % 10000 == 0:
            logger.info('{} events have been processed... Time={}ms.'.format(i, round(1000*(time.time() - start))))
    logger.info('Done! {} sequences with {} event types are found in {}ms'.format(
        seq_idx+1, type_idx+1, round(1000*(time.time() - start))))

    logger.info('Build proposed database for the sequences...')
    start2 = time.time()
    for seq_name in seq2idx.keys():
        seq_idx = seq2idx[seq_name]
        idx2seq[seq_idx] = seq_name

    for event_type in type2idx.keys():
        type_idx = type2idx[event_type]
        idx2type[type_idx] = event_type

    database['type2idx'] = type2idx
    database['idx2type'] = idx2type
    database['seq2idx'] = seq2idx
    database['idx2seq'] = idx2seq

    for i, row in df.iterrows():
        seq_name = str(row[domain_names['seq_id']])
        timestamp = float(row[domain_names['time']])
        event_type = str(row[domain_names['event']])

        seq_idx = database['seq2idx'][seq_name]
        type_idx = database['type2idx'][event_type]
        database['sequences'][seq_idx]['times'].append(timestamp)
        database['sequences'][seq_idx]['events'].append(type_idx)

        if i % 10000 == 0:
            logger.info('{} events have been processed... Time={}ms.'.format(i, round(1000*(time.time() - start2))))
    logger.info('Done! {} sequences are built in {}ms'.format(
        len(database['seq2idx']), round(1000*(time.time() - start2))))

    logger.info('Format transformation...')
    for n in range(len(database['sequences'])):
        database['sequences'][n]['t_start'] = database['sequences'][n]['times'][0]
        database['sequences'][n]['t_stop'] = database['sequences'][n]['times'][-1]+1e-2
        database['sequences'][n]['times'] = np.asarray(database['sequences'][n]['times'])
        database['sequences'][n]['events'] = np.asarray(database['sequences'][n]['events'])
        if n % 1000 == 0:
            logger.info('{} sequences have been processed... Time={}ms.'.format(n, round(1000*(time.time() - start))))
    logger.info('Done! The database has been built in {}ms'.format(round(1000*(time.time() - start))))

    return database


def load_seq_features_csv(file_name: str, seq_domain: str, domain_dict: Dict, database: Dict, normalize: int=0):
    """
    load sequences' features from a csv file
    :param file_name: the path and the name of the csv file
    :param seq_domain: the name of the key column corresponding to sequence index.
    :param domain_dict: a dictionary containing the names of the key columns corresponding to the features.
        The format should be
            domain_dict = {'domain_name': domain's feature type}
        Two types are considered:
        1) 'numerical': each element (row) in the corresponding domain should be a string containing D numbers
            separated by spaces, and D should be the same for various elements.
            D-dimensional real-value features will be generated for this domain.
            If each sequence has multiple rows, the average of the features will be recorded.

        2) 'categorical': each element (row) in the corresponding domain should be a strong containing N keywords
            separated by spaces, but N can be different for various elements.
            D-dimensional binary features will be generated for this domain. Here D is the number of distinguished
            keywords (vocabulary size).
            If each sequence has multiple rows, the aggregation of the binary features will be recorded.

    :param database: a dictionary of data generated by the function "load_sequences_csv()"
    :param normalize: 0 = no normalization, 1 = normalization across features, 2 = normalization across sequences
    :return: a database having sequences' features
    """
    df = pd.read_csv(file_name)
    num_seq = len(database['seq2idx'])
    # initialize features
    features = {}
    counts = {}
    for key in domain_dict.keys():
        features[key] = None
        counts[key] = None

    logger.info('Start to generate sequence features...')
    start = time.time()
    for i, row in df.iterrows():
        seq_name = str(row[seq_domain])
        if seq_name not in database['seq2idx'].keys():
            logger.warning("'{}' is a new sequence not appearing in current database.".format(seq_name))
            logger.warning("It will be ignored in the process.")
        else:
            seq_idx = database['seq2idx'][seq_name]
            for key in domain_dict.keys():
                elements = str(row[key])
                if domain_dict[key] == 'numerical':
                    elements = np.asarray(list(map(float, elements.split())))
                    dim = elements.shape[0]
                    if features[key] is None:
                        features[key] = np.zeros((dim, num_seq))
                        features[key][:, seq_idx] = elements
                        counts[key] = np.zeros((1, num_seq))
                        counts[key][0, seq_idx] = 1
                    else:
                        features[key][:, seq_idx] += elements
                        counts[key][0, seq_idx] += 1

                elif domain_dict[key] == 'categorical':
                    elements = elements.split()
                    if features[key] is None:
                        features[key] = {}
                        features[key][seq_idx] = elements
                        counts[key] = {}
                        element_idx = 0
                    else:
                        if seq_idx not in features[key].keys():
                            features[key][seq_idx] = elements
                        else:
                            features[key][seq_idx].extend(elements)
                    for element in elements:
                        if element not in counts[key].keys():
                            counts[key][element] = element_idx
                            element_idx += 1
                else:
                    logger.warning('Undefined feature type for the domain {}.'.format(key))
                    logger.warning("It will be ignored in the process.")
        if i % 1000 == 0:
            logger.info('{} rows have been processed... Time={}ms'.format(i, round(1000*(time.time() - start))))

    # post-process of features
    features_all = None
    start = time.time()
    for key in domain_dict.keys():
        if domain_dict[key] == 'numerical':
            features_tmp = features[key]
            features_tmp = features_tmp / np.tile(counts[key], (features[key].shape[0], 1))
            if features_all is None:
                features_all = features_tmp
            else:
                features_all = np.concatenate((features_all, features_tmp), axis=0)

        elif domain_dict[key] == 'categorical':
            features_tmp = np.zeros((len(counts[key]), num_seq))
            for seq_idx in features[key].keys():
                for element in features[key][seq_idx]:
                    element_idx = counts[key][element]
                    features_tmp[element_idx, seq_idx] += 1
            if features_all is None:
                features_all = features_tmp
            else:
                features_all = np.concatenate((features_all, features_tmp), axis=0)
        else:
            logger.warning('Undefined feature type for the domain {}.'.format(key))
            logger.warning("It will be ignored in the process.")
        logger.info("features of domain '{}' is generated... Time={}ms.".format(key, round(1000*(time.time() - start))))

    if normalize == 1:
        features_all = features_all / \
                       np.tile(np.sum(features_all, axis=0)+1e-8, (features_all.shape[0], 1))
    if normalize == 2:
        features_all = features_all / \
                       np.transpose(np.tile(np.sum(features_all, axis=1)+1e-8, (features_all.shape[1], 1)))

    for seq_idx in range(features_all.shape[1]):
        database['sequences'][seq_idx]['seq_feature'] = features_all[:, seq_idx]

    return database


def load_event_features_csv(file_name: str, event_domain: str, domain_dict: Dict, database: Dict, normalize: int=0):
    """
    load events' features from a csv file
    :param file_name: the path and the name of the csv file
    :param event_domain: the name of the key column corresponding to event index.
    :param domain_dict: a dictionary containing the names of the key columns corresponding to the features.
        The format should be
            domain_dict = {'domain_name': domain's feature type}
        Two types are considered:
        1) 'numerical': each element (row) in the corresponding domain should be a string containing D numbers
            separated by spaces, and D should be the same for various elements.
            D-dimensional real-value features will be generated for this domain.
            If each event type has multiple rows, the average of the features will be recorded.

        2) 'categorical': each element (row) in the corresponding domain should be a strong containing N keywords
            separated by spaces, but N can be different for various elements.
            D-dimensional binary features will be generated for this domain. Here D is the number of distinguished
            keywords (vocabulary size).
            If each event type has multiple rows, the aggregation of the binary features will be recorded.

    :param database: a dictionary of data generated by the function "load_sequences_csv()"
    :param normalize: 0 = no normalization, 1 = normalization across features, 2 = normalization across event types
    :return: a database having events' features
    """
    df = pd.read_csv(file_name)
    num_event = len(database['type2idx'])

    # initialize features
    features = {}
    counts = {}
    for key in domain_dict.keys():
        features[key] = None
        counts[key] = None

    logger.info('Start to generate sequence features...')
    start = time.time()
    for i, row in df.iterrows():
        event_name = str(row[event_domain])
        if event_name not in database['type2idx'].keys():
            logger.warning("'{}' is a new event type not appearing in current database.".format(event_name))
            logger.warning("It will be ignored in the process.")
        else:
            event_idx = database['type2idx'][event_name]
            for key in domain_dict.keys():
                elements = str(row[key])
                if domain_dict[key] == 'numerical':
                    elements = np.asarray(list(map(float, elements.split())))
                    dim = elements.shape[0]
                    if features[key] is None:
                        features[key] = np.zeros((dim, num_event))
                        features[key][:, event_idx] = elements
                        counts[key] = np.zeros((1, num_event))
                        counts[key][0, event_idx] = 1
                        counts[key][0, 0] = 1
                    else:
                        features[key][:, event_idx] += elements
                        counts[key][0, event_idx] += 1

                elif domain_dict[key] == 'categorical':
                    elements = elements.split()
                    if features[key] is None:
                        features[key] = {}
                        features[key][event_idx] = elements
                        counts[key] = {}
                        element_idx = 0
                    else:
                        if event_idx not in features[key].keys():
                            features[key][event_idx] = elements
                        else:
                            features[key][event_idx].extend(elements)
                    for element in elements:
                        if element not in counts[key].keys():
                            counts[key][element] = element_idx
                            element_idx += 1
                else:
                    logger.warning('Undefined feature type for the domain {}.'.format(key))
                    logger.warning("It will be ignored in the process.")
        if i % 1000 == 0:
            logger.info('{} rows have been processed... Time={}ms'.format(i, round(1000*(time.time() - start))))

    # post-process of features
    features_all = None
    start = time.time()
    for key in domain_dict.keys():
        if domain_dict[key] == 'numerical':
            features_tmp = features[key]
            features_tmp = features_tmp / np.tile(counts[key], (features[key].shape[0], 1))
            if features_all is None:
                features_all = features_tmp
            else:
                features_all = np.concatenate((features_all, features_tmp), axis=0)

        elif domain_dict[key] == 'categorical':
            features_tmp = np.zeros((len(counts[key]), num_event))
            for event_idx in features[key].keys():
                for element in features[key][event_idx]:
                    element_idx = counts[key][element]
                    features_tmp[element_idx, event_idx] += 1
            if features_all is None:
                features_all = features_tmp
            else:
                features_all = np.concatenate((features_all, features_tmp), axis=0)
        else:
            logger.warning('Undefined feature type for the domain {}.'.format(key))
            logger.warning("It will be ignored in the process.")
        logger.info("features of domain '{}' is generated... Time={}ms.".format(key, round(1000*(time.time() - start))))

    if normalize == 1:
        features_all = features_all / \
                       np.tile(np.sum(features_all, axis=0)+1e-8, (features_all.shape[0], 1))
    if normalize == 2:
        features_all = features_all / \
                       np.transpose(np.tile(np.sum(features_all, axis=1)+1e-8, (features_all.shape[1], 1)))
    database['event_features'] = features_all

    return database


def load_seq_labels_csv(file_name: str, seq_domain: str, domain_dict: Dict, database: Dict):
    """
    load sequences' features from a csv file
    :param file_name: the path and the name of the csv file
    :param seq_domain: the name of the key column corresponding to sequence index.
    :param domain_dict: a dictionary containing the name of the key column corresponding to the labels.
        The format should be
            domain_dict = {'domain_name': domain's feature type}
        The dictionary should only contain one key.
        If multiple keys are provided, only the first one is considered.

        Two types are considered:
        1) 'numerical': each element (row) in the corresponding domain should be a string containing D numbers
            separated by spaces, and D should be the same for various elements.
            D-dimensional real-value labels will be generated for this domain.
            If each sequence has multiple rows, the average of the labels will be recorded.

        2) 'categorical': each element (row) in the corresponding domain should be a strong containing N keywords.
            N-dimensional categorical label will be generated for this domain.
            If each sequence has multiple rows, the aggregation of the categories will be recorded.

    :param database: a dictionary of data generated by the function "load_sequences_csv()"
    :return: a database having sequences' labels
    """
    df = pd.read_csv(file_name)
    num_seq = len(database['seq2idx'])
    # initialize features
    keys = list(domain_dict.keys())
    label_domain = keys[0]
    if len(keys) > 1:
        logger.warning("{} label domains are found. Only the first domain '{}' is used to generate labels.".format(
            len(keys), label_domain))

    features = {}
    counts = {}
    features[label_domain] = None

    logger.info('Start to generate sequence labels...')
    start = time.time()
    for i, row in df.iterrows():
        seq_name = str(row[seq_domain])
        if seq_name not in database['seq2idx'].keys():
            logger.warning("'{}' is a new sequence not appearing in current database.".format(seq_name))
            logger.warning("It will be ignored in the process.")
        else:
            seq_idx = database['seq2idx'][seq_name]
            elements = str(row[label_domain])
            if domain_dict[label_domain] == 'numerical':
                elements = np.asarray(list(map(float, elements.split())))
                dim = elements.shape[0]
                if features[label_domain] is None:
                    features[label_domain] = np.zeros((dim, num_seq))
                    features[label_domain][:, seq_idx] = elements
                    counts[label_domain] = np.zeros((1, num_seq))
                    counts[label_domain][0, seq_idx] = 1
                else:
                    features[label_domain][:, seq_idx] += elements
                    counts[label_domain][0, seq_idx] += 1

            elif domain_dict[label_domain] == 'categorical':
                elements = elements.split()
                if features[label_domain] is None:
                    features[label_domain] = {}
                    features[label_domain][seq_idx] = elements
                    counts[label_domain] = {}
                    element_idx = 0
                else:
                    if seq_idx not in features[label_domain].keys():
                        features[label_domain][seq_idx] = elements
                    else:
                        features[label_domain][seq_idx].extend(elements)
                for element in elements:
                    if element not in counts[label_domain].keys():
                        counts[label_domain][element] = element_idx
                        element_idx += 1
            else:
                logger.warning('Undefined feature type for the domain {}.'.format(label_domain))
                logger.warning("It will be ignored in the process.")
        if i % 1000 == 0:
            logger.info('{} rows have been processed... Time={}ms.'.format(i, round(1000*(time.time() - start))))

    # post-process of features
    start = time.time()
    if domain_dict[label_domain] == 'numerical':
        features_tmp = features[label_domain]
        features_tmp = features_tmp / np.tile(counts[label_domain], (features[label_domain].shape[0], 1))
        for seq_idx in range(features_tmp.shape[1]):
            database['sequences'][seq_idx]['label'] = features_tmp[:, seq_idx]

    elif domain_dict[label_domain] == 'categorical':
        for seq_idx in features[label_domain].keys():
            elements = list(set(features[label_domain][seq_idx]))
            feature_tmp = []
            for element in elements:
                element_idx = counts[label_domain][element]
                feature_tmp.append(element_idx)
            feature_tmp = np.asarray(feature_tmp, dtype=np.int)
            database['sequences'][seq_idx]['label'] = feature_tmp
    else:
        logger.warning('Undefined label type for the domain {}.'.format(domain_dict[label_domain]))
        logger.warning("It will be ignored in the process.")
    logger.info("Labels of domain '{}' is generated... Time={}ms.".format(
        domain_dict[label_domain], round(1000*(time.time() - start))))

    return database
