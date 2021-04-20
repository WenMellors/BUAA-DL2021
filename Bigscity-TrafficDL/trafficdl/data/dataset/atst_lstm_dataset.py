import os
import json
import pandas as pd
import math
import importlib
import numpy as np
import collections
import itertools
import time
import random

from trafficdl.data.dataset import AbstractDataset
from trafficdl.data.utils import generate_dataloader


class ATSTLSTMDataset(AbstractDataset):

    def __init__(self, config):
        self.geo = None
        self.config = config
        self.cache_file_folder = './trafficdl/cache/dataset_cache/'
        self.data_path = './raw_data/gowalla/'  # 直接把数据集写定了
        self.data = None
        # 加载 encoder
        self.encoder = self.get_encoder()
        self.num_sample = config['num_sample']
        self.loc_len = 0
        self.max_dis = 0
        self.usr_num = config['usr_num']

    def get_data(self):
        """
        获取切割后的数据集
        Returns:
        """
        if self.data is None:
            if self.config['cache_dataset_load'] and os.path.exists(self.encoder.cache_file_name):
                # 加载cache
                f = open(self.encoder.cache_file_name, 'r')
                self.data = json.load(f)
                f.close()
            else:
                dyna = pd.read_csv(os.path.join(self.data_path, '{}.dyna'.format(self.config['dataset']))).values.T
                geo = pd.read_csv(os.path.join(self.data_path, '{}.geo'.format(self.config['dataset']))).values.T
                self.geo = geo
                sequence, sequence_user, sequence_time, sequence_distance, id_to_geo = self.build_sequence(dyna)
                top_n = self.pop_n(sequence, self.num_sample)
                train_set = (sequence, sequence_user, sequence_time, sequence_distance)
                final_train_set, final_eval_test, final_test_set = \
                    self.load_data(
                        train_set,
                        self.num_sample,
                        top_n,
                        id_to_geo,
                        True)
                self.data = {
                    "train": final_train_set,
                    "eval": final_eval_test,
                    "test": final_test_set
                }
                # 先不管这块
                if self.config['cache_dataset_save']:
                    if not os.path.exists(self.cache_file_folder):
                        os.makedirs(self.cache_file_folder)
                    with open(self.encoder.cache_file_name, 'w') as f:
                        json.dump(self.data, f)
        print("finish getting data!")
        # 这里没传self.pad_item
        return generate_dataloader(train_data=self.data["train"],
                                   eval_data=self.data["eval"],
                                   test_data=self.data["test"],
                                   feature_name={'loc': 'float', 'tim': 'float', 'dis': 'float', 'uid': 'float',
                                                 'loc_neg': 'float', 'tim_neg': 'float', 'dis_neg': 'float',
                                                 'target': 'int'},
                                   batch_size=self.config['batch_size'],
                                   num_workers=self.config['num_workers'],
                                   shuffle=False
                                   )

    def get_data_feature(self):
        """
        所有数据都在get_data里分析好了，这部分仅留接口
        我看了他的data_feature传给模型了，但是不知道用来做什么。不管做什么，模型需要的data应该都从get_data中能得到。有问题我再修改
        Returns:None
        """
        return None

    def get_geo(self, location):
        """
        获取地理信息
        Args:
            location: 兴趣点标号

        Returns:
            if (兴趣点标号合法）
                return 兴趣点对应物理坐标（经纬度）
            else
                return [-1,-1]
        """
        left = 0
        right = self.geo.shape[1] - 1
        while left <= right:  # 循环条件
            mid = (left + right) // 2  # 获取中间位置，数字的索引（序列前提是有序的）
            if location < self.geo[0][mid]:  # 如果查询数字比中间数字小，那就去二分后的左边找，
                right = mid - 1  # 来到左边后，需要将右变的边界换为mid-1
            elif location > self.geo[0][mid]:  # 如果查询数字比中间数字大，那么去二分后的右边找
                left = mid + 1  # 来到右边后，需要将左边的边界换为mid+1
            else:
                [x, y] = self.geo[2, mid][1:-1].split(',')  # 如果查询数字刚好为中间值，返回该值得索引
                return [float(x), float(y)]
        return [-1, -1]

    def get_encoder(self):
        try:
            return getattr(importlib.import_module('trafficdl.data.dataset.trajectory_encoder'),
                           self.config['traj_encoder'])(self.config)
        except AttributeError:
            raise AttributeError('trajectory encoder is not found')

    def build_sequence(self, dyna):
        user_voc = collections.Counter(list(dyna[3, :]))
        sequence = []
        sequence_user = []
        sequence_time = []
        sequence_distance = []

        print("building sequence...")
        k = 0
        sum_sequence = 0
        usr_num = len(user_voc.keys())
        max_usr = sorted(dyna[3, :])[-1]
        user_voc_set = set(user_voc.keys())
        while user_voc_set:
            user = random.sample(user_voc_set, 1)[0]
            k = k + 1
            if k % self.usr_num == 0:
                print("finish loading {}/{} users out of {} users".format(k, self.usr_num, usr_num))
                break
            if k % 200 == 0:
                print("finish loading {}/{} users out of {} users".format(k, self.usr_num, usr_num))
            # 找到该用户所有check-in
            checkin_user_redex = np.argwhere(dyna[3, :] == user)
            checkin_user_all = dyna[:, checkin_user_redex[:, 0]]

            user_count = 0
            sequence_location = []
            sequence_time_user = []
            sequence_distance_user = []

            temperal_sequence_location = []
            temperal_sequence_time_user = []
            temperal_sequence_distance_user = []

            # 对该用户的check-in按发生时间顺序排序
            sorted_time = np.sort(checkin_user_all[2, :])
            # 返回对应索引
            sorted_time_index = np.argsort(checkin_user_all[2, :])

            ''' dyna
            2 : time
            3 : id
            4 : location
            '''
            for i in range(len(checkin_user_redex)):
                if i == 0:
                    sequence_location.append(checkin_user_all[4, sorted_time_index[i]])
                    sequence_time_user.append(100)
                    sequence_distance_user.append(1)  # 物理距离设为1
                else:
                    timeArray_front = time.strptime(sorted_time[i - 1], "%Y-%m-%dT%H:%M:%SZ")
                    timeArray_behind = time.strptime(sorted_time[i], "%Y-%m-%dT%H:%M:%SZ")
                    delta_time = int(time.mktime(timeArray_behind)) - int(time.mktime(timeArray_front))
                    if delta_time > 21600:  # 两端时间间隔长于21600
                        if len(sequence_location) > 4:  # 够长就化为另一个用户存起来，用户数++
                            sequence_location = list(map(int, sequence_location))
                            sequence_time_user = list(map(int, sequence_time_user))
                            temperal_sequence_location.append(sequence_location)
                            temperal_sequence_time_user.append(sequence_time_user)
                            temperal_sequence_distance_user.append(sequence_distance_user)
                            user_count = user_count + 1

                        sequence_location = []  # 清空
                        sequence_time_user = []
                        sequence_distance_user = []
                        sequence_location.append(checkin_user_all[4, sorted_time_index[i]])
                        sequence_time_user.append(100)
                        sequence_distance_user.append(1)
                    else:
                        sequence_location.append(checkin_user_all[4, sorted_time_index[i]])
                        sequence_time_user.append(delta_time + 1e-5)  # 这里记的是时间间隔（为啥要1e-5）
                        # 经纬度序列（并不确定哪个是经度哪个是维度）
                        [latitude, longitude] = self.get_geo(
                            checkin_user_all[4, sorted_time_index[i]])
                        [prev_la, prev_lo] = self.get_geo(
                            checkin_user_all[4, sorted_time_index[i - 1]])
                        distance = self.haversine((latitude, longitude), (prev_la, prev_lo))
                        sequence_distance_user.append(distance + 1e-5)  # 和上一个点的物理距离
            sum_sequence = sum_sequence + user_count

            if user_count > 5:  # 如果该用户有超过5个序列，才记录
                sequence = sequence + temperal_sequence_location
                sequence_time = sequence_time + temperal_sequence_time_user
                sequence_distance = sequence_distance + temperal_sequence_distance_user
                sequence_user = sequence_user + [user / max_usr] * user_count

            user_voc_set.remove(user)

        print("renumbering data...")
        max_time = max([max(x) for x in sequence_time])
        self.max_dis = max([max(x) for x in sequence_distance])
        sequence_time = [[y / max_time for y in x] for x in sequence_time]  # 以最大时间间隔为分母
        sequence_distance = [[y / self.max_dis for y in x] for x in sequence_distance]  # 以最大距离间隔为分母

        # 把原来的位置信息换成01234
        Locations_voc = collections.Counter(list(itertools.chain.from_iterable(sequence)))
        location_list = list(Locations_voc.keys())
        self.loc_len = len(location_list)
        newsequence = []
        word_to_id = dict(zip(location_list, range(len(location_list))))
        id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
        id_to_geo = [self.get_geo(id_to_word[x]) for x in range(len(id_to_word))]

        for lst in sequence:
            newsequence.append([word_to_id[x] / self.loc_len for x in lst])

        # 返回location，每个sequence对应的用户号，和前一个的间隔时间，和前一个间隔距离
        return newsequence, sequence_user, sequence_time, sequence_distance, id_to_geo

    def haversine(self, lonlat1, lonlat2):
        """
        Args:
            lonlat1: 第一个点的物理坐标
            lonlat2: 第二个点的物理坐标

        Returns:
            两个点之间的物理距离
        """
        lat1, lon1 = lonlat1
        lat2, lon2 = lonlat2
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371
        return c * r

    def pop_n(self, sequence, k):
        """
        得到出现频率最高的n个序列
        Args:
            sequence:序列集
            k: n

        Returns:
            n个序列
        """
        Locations_voc = collections.Counter(list(itertools.chain.from_iterable(sequence)))
        sorted_Locations_voc = sorted(Locations_voc.items(), key=lambda d: d[1], reverse=True)
        return [a for i, (a, b) in enumerate(sorted_Locations_voc) if i < k]

    def load_data(self, train_set, num_sample, top_n, id_to_geo, sort_by_len=True):
        """
        得到最终数据
        Args:
            train_set: 训练集
            num_sample: negetive sample数
            top_n: 前500个出现频率最高的序列
            id_to_geo: 由id得到物理信息
            sort_by_len: 是否按序列长度排序

        Returns:
            train：
            eval：
            test：
            negetive
        """
        (train_set_sequence, sequence_user, train_set_time, train_set_distance) = train_set

        max_len = max([len(x) for x in train_set_sequence])
        new_sequence = []
        new_sequence_user = []
        new_time = []
        new_distance = []

        print("data augmenting...")
        # data augmentation [01234]->[012][0123][01234] 这块应该只有idx
        for k in range(len(train_set_sequence)):
            for i in range(len(train_set_sequence[k]) - 2):
                new_sequence.append(train_set_sequence[k][0:i + 3])
                new_sequence_user.append(sequence_user[k])
                new_time.append(train_set_time[k][0:i + 3])
                new_distance.append(train_set_distance[k][0:i + 3])

        print("generating train/eval/test...")
        n_samples = len(new_sequence)
        # 顺序打乱
        sidx = np.random.permutation(n_samples)

        train_rate = self.config['train_rate']
        eval_rate = self.config['eval_rate']

        n_train = int(np.round(n_samples * train_rate))
        n_eval = int(np.round(n_samples * eval_rate))

        # train
        train_set_sequence = [new_sequence[s] for s in sidx[:n_train]]
        train_set_time = [new_time[s] for s in sidx[:n_train]]
        train_set_distance = [new_distance[s] for s in sidx[:n_train]]
        train_set_user = [new_sequence_user[s] for s in sidx[:n_train]]
        # eval
        eval_set_sequence = [new_sequence[s] for s in sidx[n_train:n_train + n_eval]]
        eval_set_time = [new_time[s] for s in sidx[n_train:n_train + n_eval]]
        eval_set_distance = [new_distance[s] for s in sidx[n_train:n_train + n_eval]]
        eval_set_user = [new_sequence_user[s] for s in sidx[n_train:n_train + n_eval]]
        # test
        test_set_sequence = [new_sequence[s] for s in sidx[n_train + n_eval:]]
        test_set_time = [new_time[s] for s in sidx[n_train + n_eval:]]
        test_set_distance = [new_distance[s] for s in sidx[n_train + n_eval:]]
        test_set_user = [new_sequence_user[s] for s in sidx[n_train + n_eval:]]

        def sort_by_length(seq, time, distance, usr):
            sorted_index = sorted(range(len(seq)), key=lambda x: len(seq[x]))
            set_sequence = [seq[i] for i in sorted_index]
            set_time = [time[i] for i in sorted_index]
            set_distance = [distance[i] for i in sorted_index]
            set_user = [usr[i] for i in sorted_index]
            return set_sequence, set_time, set_distance, set_user

        if sort_by_len:
            test_set_sequence, test_set_time, test_set_distance, test_set_user = sort_by_length(
                test_set_sequence, test_set_time, test_set_distance, test_set_user
            )
            eval_set_sequence, eval_set_time, eval_set_distance, eval_set_user = sort_by_length(
                eval_set_sequence, eval_set_time, eval_set_distance, eval_set_user
            )
            train_set_sequence, train_set_time, train_set_distance, train_set_user = sort_by_length(
                train_set_sequence, train_set_time, train_set_distance, train_set_user
            )

        # 前面的作为判据，最后一列作为结果
        def divide_data(seq, time, distance):
            set_sequence_x = [x[0:len(x) - 1] for x in seq]
            set_time_x = [x[0:len(x) - 1] for x in time]
            set_distance_x = [x[0:len(x) - 1] for x in distance]
            set_sequence_y = [x[len(x) - 1] for x in seq]
            set_time_y = [x[len(x) - 1] for x in time]
            set_distance_y = [x[len(x) - 1] for x in distance]
            new_sequence_x = np.zeros([len(set_sequence_x), max_len])
            new_time_x = np.zeros([len(set_time_x), max_len])
            new_distance_x = np.zeros([len(set_distance_x), max_len])
            new_sequence_y = np.zeros([len(set_sequence_y), 1])
            new_time_y = np.zeros([len(set_time_y), 1])
            new_distance_y = np.zeros([len(set_distance_y), 1])
            return set_sequence_x, set_time_x, set_distance_x, set_sequence_y, set_time_y, set_distance_y \
                , new_sequence_x, new_time_x, new_distance_x, new_sequence_y, new_time_y, new_distance_y

        train_sequence_x, train_time_x, train_distance_x, train_sequence_y, train_time_y, train_distance_y, \
        new_train_sequence_x, new_train_time_x, new_train_distance_x, new_train_sequence_y, new_train_time_y, \
        new_train_distance_y = divide_data(train_set_sequence, train_set_time, train_set_distance)

        eval_sequence_x, eval_time_x, eval_distance_x, eval_sequence_y, eval_time_y, eval_distance_y, \
        new_eval_sequence_x, new_eval_time_x, new_eval_distance_x, new_eval_sequence_y, new_eval_time_y, \
        new_eval_distance_y = divide_data(eval_set_sequence, eval_set_time, eval_set_distance)

        test_sequence_x, test_time_x, test_distance_x, test_sequence_y, test_time_y, test_distance_y, \
        new_test_sequence_x, new_test_time_x, new_test_distance_x, new_test_sequence_y, new_test_time_y, \
        new_test_distance_y = divide_data(test_set_sequence, test_set_time, test_set_distance)

        negative_train = np.zeros([len(new_train_sequence_y), num_sample])
        negative_time_train = np.zeros([len(new_train_sequence_y), num_sample])
        negative_distance_train = np.zeros([len(new_train_sequence_y), num_sample])

        negative_eval = np.zeros([len(new_eval_sequence_y), num_sample])
        negative_time_eval = np.zeros([len(new_eval_sequence_y), num_sample])
        negative_distance_eval = np.zeros([len(new_eval_sequence_y), num_sample])

        negative_test = np.zeros([len(new_test_sequence_y), num_sample])
        negative_time_test = np.zeros([len(new_test_sequence_y), num_sample])
        negative_distance_test = np.zeros([len(new_test_sequence_y), num_sample])

        print("padding data...")

        # 不够的补齐，够长的截断
        def padding(x, y, new_x, new_y, max_len):
            (data_num, data_len) = new_x.shape
            for i in range(data_num):
                if len(x[i]) <= max_len:
                    # 不够长的应该是0补齐？
                    new_x[i, 0:len(x[i])] = x[i]
                else:
                    # 超长的截断
                    new_x[i] = x[i, 0:max_len]
            new_x = np.column_stack((new_x, y))
            return new_x

        new_train_set_sequence = padding(train_sequence_x, train_sequence_y,
                                         new_train_sequence_x, new_train_sequence_y, max_len)
        new_train_set_time = padding(train_time_x, train_time_y,
                                     new_train_time_x, new_train_time_y, max_len)
        new_train_set_distance = padding(train_distance_x, train_distance_y,
                                         new_train_distance_x, new_train_distance_y, max_len)

        new_eval_set_sequence = padding(eval_sequence_x, eval_sequence_y,
                                        new_eval_sequence_x, new_eval_sequence_y, max_len)
        new_eval_set_time = padding(eval_time_x, eval_time_y,
                                    new_eval_time_x, new_eval_time_y, max_len)
        new_eval_set_distance = padding(eval_distance_x, eval_distance_y,
                                        new_eval_distance_x, new_eval_distance_y, max_len)

        new_test_set_sequence = padding(test_sequence_x, test_sequence_y,
                                        new_test_sequence_x, new_test_sequence_y, max_len)
        new_test_set_time = padding(test_time_x, test_time_y,
                                    new_test_time_x, new_test_time_y, max_len)
        new_test_set_distance = padding(test_distance_x, test_distance_y,
                                        new_test_distance_x, new_test_distance_y, max_len)

        def padding_negative_sample(targets, negative_sample, negative_distance_sample, top_n):
            suqence_num, num_sample = negative_sample.shape
            # 把id_to_geo做好了，这块把循环换成top_500的geo和target的geo的计算距离
            for i in range(suqence_num):
                target_geo = id_to_geo[int(targets[i][-1] * self.loc_len)]
                negative_sample[i] = np.array(top_n)
                for j in range(num_sample):
                    c_geo = id_to_geo[int(top_n[j] * self.loc_len)]
                    negative_distance_sample[i, j] = self.haversine(target_geo, c_geo) / self.max_dis
            return negative_sample, negative_distance_sample

        negative_samples_train, negative_distance_samples_train = padding_negative_sample(train_sequence_x,
                                                                                          negative_train,
                                                                                          negative_distance_train,
                                                                                          top_n)

        negative_samples_eval, negative_distance_samples_eval = padding_negative_sample(eval_sequence_x, negative_eval,
                                                                                        negative_distance_eval,
                                                                                        top_n)

        negative_samples_test, negative_distance_samples_test = padding_negative_sample(test_sequence_x, negative_test,
                                                                                        negative_distance_test,
                                                                                        top_n)

        for i in range(num_sample):
            negative_time_train[:, i] = train_time_y
            negative_time_eval[:, i] = eval_time_y
            negative_time_test[:, i] = test_time_y

        def to_input(seq_x, time_x, dis_x, usr_x, neg_s, neg_t, neg_d):
            set = []
            for i in range(len(seq_x)):
                input = [list(seq_x[i]), list(time_x[i]), list(dis_x[i]), usr_x[i],
                         list(neg_s[i]), list(neg_t[i]), list(neg_d[i]), 0]
                set.append(input)
            return set

        final_train_set = to_input(
            new_train_set_sequence, new_train_set_distance, new_train_set_time, train_set_user,
            negative_samples_train, negative_time_train, negative_distance_samples_train)
        final_eval_set = to_input(
            new_eval_set_sequence, new_eval_set_distance, new_eval_set_time, eval_set_user,
            negative_samples_eval, negative_time_eval, negative_distance_samples_eval)
        final_test_set = to_input(
            new_test_set_sequence, new_test_set_distance, new_test_set_time, test_set_user,
            negative_samples_test, negative_time_test, negative_distance_samples_test)

        return final_train_set, final_eval_set, final_test_set
