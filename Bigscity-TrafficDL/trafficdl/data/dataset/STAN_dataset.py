import os
import json
import pandas as pd
import math
import importlib
import json
from datetime import datetime

from trafficdl.data.dataset import AbstractDataset
from trafficdl.utils import parse_time, cal_basetime, cal_timeoff
from trafficdl.data.utils import generate_dataloader
import numpy as np
import torch
from math import radians, cos, sin, asin, sqrt
import joblib
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data
from tqdm import tqdm
import pdb


hours = 24*7


def strlist2list(strlist: str):
    return json.loads(strlist)

def convert_geo(geo: pd.DataFrame):
    def convert_line(line: np.ndarray):
        _id = line[0] + 1  
        _x, _y = strlist2list(line[1])
        return float(_id), float(_x), float(_y)
    print('>>> Converting geo...')
    newgeo = [convert_line(line) for line in tqdm(geo.values)]
    newgeo_np = np.array(newgeo)
    return newgeo_np

def convert_traj(traj: pd.DataFrame):
    def convert_line(line: np.ndarray):
        entity_id = line[0] + 1  # load.py gives id from 1
        location = line[1] + 1  # load.py gives id from 1
        time = datetime.strptime(line[2], "%Y-%m-%dT%H:%M:%SZ")
        timestamp = time.timestamp()
        return int(entity_id), int(location), int(timestamp)
    print('>>> Converting traj...')
    traj = traj.sort_values(by='entity_id')
    newtraj = [convert_line(line) for line in tqdm(traj.values)]
    newtraj_np = np.array(newtraj)
    return newtraj_np

class LocalDataset(data.Dataset):
    def __init__(self, traj, m1, v, label, length, device):
        # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M), (NUM), (NUM)
        self.traj, self.mat1, self.vec, self.label, self.length = traj, m1, v, label, length
        self.device = device

    def __getitem__(self, index):
        traj = self.traj[index].to(self.device)
        mats1 = self.mat1[index].to(self.device)
        vector = self.vec[index].to(self.device)
        label = self.label[index].to(self.device)
        length = self.length[index].to(self.device)
        return traj, mats1, vector, label, length

    def __len__(self):  # no use
        return len(self.traj)


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


def euclidean(point, each):
    lon1, lat1, lon2, lat2 = point[2], point[1], each[2], each[1]
    return np.sqrt((lon1 - lon2)**2 + (lat1 - lat2)**2)


def rst_mat1(traj, poi):
    # traj (*M, [u, l, t]), poi(L, [l, lat, lon])
    mat = np.zeros((len(traj), len(traj), 2))
    for i, item in enumerate(traj):
        for j, term in enumerate(traj):
            poi_item, poi_term = poi[item[1] - 1], poi[term[1] - 1]  # retrieve poi by loc_id
            mat[i, j, 0] = haversine(lon1=poi_item[2], lat1=poi_item[1], lon2=poi_term[2], lat2=poi_term[1])
            mat[i, j, 1] = abs(item[2] - term[2])
    return mat  # (*M, *M, [dis, tim])


def rs_mat2s(poi, l_max):
    # poi(L, [l, lat, lon])
    candidate_loc = np.linspace(1, l_max, l_max)  # (L)
    mat = np.zeros((l_max, l_max))  # mat (L, L)
    for i, loc1 in enumerate(tqdm(candidate_loc)):
        # print(i) if i % 100 == 0 else None
        for j, loc2 in enumerate(candidate_loc):
            poi1, poi2 = poi[int(loc1) - 1], poi[int(loc2) - 1]  # retrieve poi by loc_id
            mat[i, j] = haversine(lon1=poi1[2], lat1=poi1[1], lon2=poi2[2], lat2=poi2[1])
    return mat  # (L, L)


def rt_mat2t(traj_time):  # traj_time (*M+1) triangle matrix
    # construct a list of relative times w.r.t. causality
    mat = np.zeros((len(traj_time)-1, len(traj_time)-1))
    for i, item in enumerate(traj_time):  # label
        if i == 0:
            continue
        for j, term in enumerate(traj_time[:i]):  # data
            mat[i-1, j] = np.abs(item - term)
    return mat  # (*M, *M)


class StanTrajectoryDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.cache_file_folder = './trafficdl/cache/dataset_cache/'
        self.data_path = './raw_data/{}/'.format(self.config['dataset'])
        self.batch_size = 4 # N = 1

        # define cache data filename
        cache_file = os.path.join(self.cache_file_folder, self.config['dataset'] + '_data.pkl')
        self.file_data = self._prepare_data(cache_file)
        
        # tensor(NUM, M, 3), np(NUM, M, M, 2), np(L, L), np(NUM, M, M), tensor(NUM, M), np(NUM)
        [trajs, mat1, mat2s, mat2t, labels, lens, u_max, l_max] = self.file_data
        mat1, mat2s, mat2t, lens = torch.FloatTensor(mat1), torch.FloatTensor(mat2s).to(self.device), \
                                torch.FloatTensor(mat2t), torch.LongTensor(lens)

        # the run speed is very flow due to the use of location matrix (also huge memory cost)
        # please use a partition of the data (recommended)
        part = 100
        trajs, mat1, mat2t, labels, lens = \
            trajs[:part], mat1[:part], mat2t[:part], labels[:part], lens[:part]
        ex = mat1[:, :, :, 0].max(), mat1[:, :, :, 0].min(), mat1[:, :, :, 1].max(), mat1[:, :, :, 1].min()
        self.data_feature = {'tim_size':hours + 1,
                             'loc_size':l_max + 1,
                             'uid_size':u_max + 1,
                             'ex': ex,
                             'mat2s':mat2s}

        self.dataset = LocalDataset(trajs, mat1, mat2t, labels-1, lens, self.device)
        self.data_loader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

    def _prepare_data(self, cache_file: str):
        if not os.path.exists(cache_file):
        # if True:
            _geo = pd.read_csv(os.path.join(self.data_path, '{}.geo'.format(self.config['dataset'])))
            poi = convert_geo(_geo[['geo_id', 'coordinates']])
            _traj = pd.read_csv(os.path.join(self.data_path, '{}.dyna'.format(self.config['dataset'])))
            traj = convert_traj(_traj[['entity_id', 'location', 'time']])
            print('>>> Building cache from scratch in _process_traj...')
            self._process_traj(cache_file, traj, poi, self.config['max_len'])
        
        assert os.path.exists(cache_file), "Error in data caching system!"
        with open(cache_file, 'rb') as cache_fp:
            print('>>> Loading cached data file...')
            ret = joblib.load(cache_fp)
        print('>>> File data prepared.')
        return ret
    
    def _process_traj(self, data_pkl: str, data: np.ndarray, poi: np.ndarray, max_len: int):  # start from 1
        # data (?, [u, l, t]), poi (L, [l, lat, lon])
        # data = np.load('./data/' + dname + '.npy')
        # poi = np.load('./data/' + dname + '_POI.npy')
        num_user = data[-1, 0]  # max id of users, i.e. NUM
        data_user = data[:, 0]  # user_id sequence in data
        trajs, labels, mat1, mat2t, lens = [], [], [], [], []
        u_max, l_max = np.max(data[:, 0]), np.max(data[:, 1])
        pdb.set_trace()
        print('>>> Dispatching data from input...')
        for u_id in tqdm(range(num_user+1)):
            if u_id == 0:  # skip u_id == 0
                continue
            init_mat1 = np.zeros((max_len, max_len, 2))  # first mat (M, M, 2)
            init_mat2t = np.zeros((max_len, max_len))  # second mat of time (M, M)
            user_traj = data[np.where(data_user == u_id)]  # find all check-ins of u_id
            user_traj = user_traj[np.argsort(user_traj[:, 2])].copy()  # sort traj by time

            # print(u_id, len(user_traj)) if u_id % 100 == 0 else None

            if len(user_traj) > max_len + 1:  # consider only the M+1 recent check-ins
                # 0:-3 are training data, 1:-2 is training label;
                # 1:-2 are validation data, 2:-1 is validation label;
                # 2:-1 are test data, 3: is the label for test.
                # *M would be the real length if <= max_len + 1
                user_traj = user_traj[-max_len-1:]  # (*M+1, [u, l, t])

            # spatial and temporal intervals
            user_len = len(user_traj[:-1])  # the len of data, i.e. *M
            user_mat1 = rst_mat1(user_traj[:-1], poi)  # (*M, *M, [dis, tim])
            user_mat2t = rt_mat2t(user_traj[:, 2])  # (*M, *M)
            init_mat1[0:user_len, 0:user_len] = user_mat1
            init_mat2t[0:user_len, 0:user_len] = user_mat2t

            trajs.append(torch.LongTensor(user_traj)[:-1])  # (NUM, *M, [u, l, t])
            mat1.append(init_mat1)  # (NUM, M, M, 2)
            mat2t.append(init_mat2t)  # (NUM, M, M)
            labels.append(torch.LongTensor(user_traj[1:, 1]))  # (NUM, *M)
            lens.append(user_len-2)  # (NUM), the real *M for every user

        # padding zero to the vacancies in the right
        print('>>> Building distance of all locations...')
        mat2s = rs_mat2s(poi, l_max)  # contains dis of all locations, (L, L)
        print('>>> Sorting dispatched data...')
        zipped = zip(*sorted(zip(trajs, mat1, mat2t, labels, lens), key=lambda x: len(x[0]), reverse=True))
        trajs, mat1, mat2t, labels, lens = zipped
        trajs, mat1, mat2t, labels, lens = list(trajs), list(mat1), list(mat2t), list(labels), list(lens)
        print('>>> Padding sequence...')
        trajs = pad_sequence(trajs, batch_first=True, padding_value=0)  # (NUM, M, 3)
        labels = pad_sequence(labels, batch_first=True, padding_value=0)  # (NUM, M)
        print('>>> Done padding.')
        data = [trajs, np.array(mat1), mat2s, np.array(mat2t), labels, np.array(lens), u_max, l_max]

        with open(data_pkl, 'wb') as pkl:
            joblib.dump(data, pkl)
            print('>>> Successfully dump to cache.')

    def get_data(self):
        # train, eval, test
        return self.data_loader, null, null

    def get_data_feature(self):
        return self.data_feature


if __name__ == '__main__':
    import sys
    config = {
        'dataset': sys.argv[1],
         'device': 'cuda',
         'max_len': 100,
    }
    dataset = StanTrajectoryDataset(config=config)
    print('SUCCESS!')
    pass
