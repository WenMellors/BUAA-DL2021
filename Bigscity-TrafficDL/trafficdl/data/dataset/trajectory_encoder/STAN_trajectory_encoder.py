import os

from trafficdl.data.dataset.trajectory_encoder.abstract_trajectory_encoder import AbstractTrajectoryEncoder
from trafficdl.utils import parse_time, cal_basetime, cal_timeoff

import numpy as np
from math import radians, cos, sin, asin, sqrt

max_len = 100  # max traj len; i.e., M
parameter_list = ['dataset', 'min_session_len', 'min_sessions', 'traj_encoder', 'window_type',
                  'window_size', 'history_type']


# from STAN/load.py
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
    for i, loc1 in enumerate(candidate_loc):
        print(i) if i % 100 == 0 else None
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


class STANTrajectoryEncoder(AbstractTrajectoryEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.uid = 0
        self.location2id = {}  # 因为原始数据集中的部分 loc id 不会被使用到因此这里需要重新编码一下
        self.loc_id = 0
        self.tim_max = 0  # 记录最大的时间编码
        if self.config['window_type'] == 'time_window':
            # 对于以时间窗口切割的轨迹，最大时间编码是已知的
            self.tim_max = self.config['window_size'] - 1
        self.history_type = self.config['history_type']
        self.feature_dict = {'history_loc': 'int', 'history_tim': 'int',
                             'current_loc': 'int', 'current_tim': 'int',
                             'target': 'int', 'target_tim': 'int', 'uid': 'int'
                             }
        self.feature_max_len = {
            'history_loc': self.config['history_len'],
            'history_tim': self.config['history_len']
        }
        parameters_str = ''
        for key in parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './trafficdl/cache/dataset_cache/', 'trajectory_{}.json'.format(parameters_str))
        # 对于这种 history 模式没办法做到 batch
        if self.history_type == 'cut_off':
            # self.config['batch_size'] = 1
            self.feature_name['history_loc'] = 'array of int'
            self.feature_name['history_tim'] = 'array of int'

    def encode(self, uid, trajectories):
        """standard encoder use the same method as DeepMove

        Recode poi id. Encode timestamp with its hour.

        Args:
            uid ([type]): same as AbstractTrajectoryEncoder
            trajectories ([type]): same as AbstractTrajectoryEncoder
                trajectory1 = [
                    (location ID, timestamp, timezone_offset_in_minutes),
                    (location ID, timestamp, timezone_offset_in_minutes),
                    .....
                ]
        """
        # 直接对 uid 进行重编码
        uid = self.uid
        self.uid += 1
        encoded_trajectories = []
        history_loc = []
        history_tim = []
        for index, traj in enumerate(trajectories):
            current_loc = []
            current_tim = []
            start_time = parse_time(traj[0][1], traj[0][2])
            # 以当天凌晨的时间作为计算 time_off 的基准
            base_time = cal_basetime(start_time, True)
            for point in traj:
                loc = point[0]
                now_time = parse_time(point[1], point[2])
                if loc not in self.location2id:
                    self.location2id[loc] = self.loc_id
                    self.loc_id += 1
                current_loc.append(self.location2id[loc])
                time_code = int(cal_timeoff(now_time, base_time))
                if time_code > self.tim_max:
                    self.tim_max = time_code
                current_tim.append(time_code)
            # 完成当前轨迹的编码，下面进行输入的形成
            if index == 0:
                # 因为要历史轨迹特征，所以第一条轨迹是不能构成模型输入的
                if self.history_type == 'splice':
                    history_loc += current_loc
                    history_tim += current_tim
                else:
                    history_loc.append(current_loc)
                    history_tim.append(current_tim)
                continue
            trace = []
            target = current_loc[-1]
            target_tim = current_tim[-1]
            current_loc = current_loc[:-1]
            current_tim = current_tim[:-1]
            trace.append(history_loc)
            trace.append(history_tim)
            trace.append(current_loc)
            trace.append(current_tim)
            trace.append(target)
            trace.append(target_tim)
            trace.append(uid)
            encoded_trajectories.append(trace)
            if self.history_type == 'splice':
                history_loc += current_loc
                history_tim += current_tim
            else:
                history_loc.append(current_loc)
                history_tim.append(current_tim)
        return encoded_trajectories

    def gen_data_feature(self):
        loc_pad = self.loc_id
        tim_pad = self.tim_max + 1
        if self.history_type == 'cut_off':
            self.pad_item = {
                'current_loc': loc_pad,
                'current_tim': tim_pad
            }
            # 这种情况下不对 history_loc history_tim 做补齐
        else:
            self.pad_item = {
                'current_loc': loc_pad,
                'history_loc': loc_pad,
                'current_tim': tim_pad,
                'history_tim': tim_pad
            }
        self.data_feature = {
            'loc_size': self.loc_id + 1,
            'tim_size': self.tim_max + 2,
            'uid_size': self.uid,
            'loc_pad': loc_pad,
            'tim_pad': tim_pad
        }
