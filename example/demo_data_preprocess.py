"""
Test different data operations.
"""

import dev.util as util
from preprocess.DataIO import load_sequences_csv, load_seq_features_csv, load_event_features_csv
from preprocess.DataOperation import stitching, superposing, aggregating, data_info

# test sequence loading functions
# load event sequences
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
                                 database=database)
data_info(database)

# load event types' features
database = load_event_features_csv('{}/{}/Linkedin.csv'.format(util.POPPY_PATH, util.DATA_DIR),
                                   event_domain='event',
                                   domain_dict=domain_dict,
                                   database=database)
data_info(database)

# test data operators
# stitching
database1 = stitching(database, database, method='random')
data_info(database1)

database2 = stitching(database, database, method='feature')
# superposing
database3 = superposing(database, database, method='random')
database4 = superposing(database, database, method='feature')
# aggregating
database5 = aggregating(database, dt=1)
