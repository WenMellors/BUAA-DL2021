import os
import torch
from trafficdl.data.dataset.trajectory_encoder.abstract_trajectory_encoder import AbstractTrajectoryEncoder
from trafficdl.utils import parse_time, cal_basetime, cal_timeoff

parameter_list = ['dataset', 'min_session_len', 'min_sessions', 'traj_encoder', 'window_type',
                  'window_size', 'history_type']


class STRNNTrajectoryEncoder(AbstractTrajectoryEncoder):

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

        # f_geo = open('D:\\2021春\深度学习\中作业\BUAA-DL2021\Bigscity-TrafficDL\\raw_data\gowalla\gowalla.geo', 'r')
        f_geo = open('E:\\深度学习\final\BUAA-DL2021\BUAA-DL2021\Bigscity-TrafficDL\\raw_data\\foursquare_tky\\foursquare_tky.geo', 'r')
        lines = f_geo.readlines()
        geo_coord = {}
        for i, line in enumerate(lines):
            if i == 0:
                continue
            tokens = line.strip().replace("\"", "").replace("[", "").replace("]", "").split(',')
            # print(tokens)
            loc_id, loc_longi, loc_lati = int(tokens[0]), tokens[2], tokens[3]
            geo_coord[loc_id] = [loc_lati, loc_longi]
        f_geo.close()
        # print(len(geo_coord))
        # print(geo_coord[816])

        for index, traj in enumerate(trajectories):
            current_loc = []
            current_dis = []
            current_longi = []
            current_lati = []
            current_tim = []
            start_time = parse_time(traj[0][1], traj[0][2])
            # 以当天凌晨的时间作为计算 time_off 的基准
            base_time = cal_basetime(start_time, True)
            for point in traj:
                loc = point[0]
                now_time = parse_time(point[1], point[2])
                if loc not in self.location2id:
                    # self.location2id[loc] = self.loc_id
                    self.location2id[loc] = loc
                    self.loc_id += 1
                current_loc.append(self.location2id[loc])
                current_lati.append(geo_coord[loc][0])
                current_longi.append(geo_coord[loc][1])
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
            lati = geo_coord[self.location2id[current_loc]][0]
            longi = geo_coord[self.location2id[current_loc]][0]
            current_dis = euclidean_dist(lati-current_lati, longi-current_longi)
            current_tim = current_tim[:-1]
            trace.append(current_dis)
            trace.append(history_loc)
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

def euclidean_dist(x, y):
    return torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))