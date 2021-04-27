import os
import json
import pandas as pd
import math
import importlib

from trafficdl.data.dataset import AbstractDataset
from trafficdl.utils import parse_time, cal_basetime, cal_timeoff
from trafficdl.data.utils import generate_dataloader


class STFRNNDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        self.cache_file_folder = './trafficdl/cache/dataset_cache/'
        self.data_path = './raw_data/{}/'.format(self.config['dataset'])
        self.data = None

    def get_data(self):

        n = len(self.data)
        train_data, eval_data, test_data = self.data[:int(n*0.8)], \
                                           self.data[int(n*0.8): int(n*0.9)], \
                                           self.data[int(n * 0.9):]
        return generate_dataloader(train_data, eval_data, test_data,
                                   {
                                       "latitude" : "float",
                                       "longitude" : "float",
                                       "date" : "emm",
                                       "time" : "emm",
                                   },
                                   self.config['batch_size'],
                                   self.config['num_workers'])

    def get_data_feature(self):
        # res = self.data['data_feature']
        # # 由 dataset 添加 poi_profile
        # res['poi_profile'] = pd.read_csv(os.path.join(
        #     self.data_path, '{}.geo'.format(self.config['dataset'])))
        # with open(os.path.join(self.data_path, 'config.json'), 'r') as f:
        #     config = json.load(f)
        #     res['distance_upper'] = config['info']['distance_upper']
        return {}

