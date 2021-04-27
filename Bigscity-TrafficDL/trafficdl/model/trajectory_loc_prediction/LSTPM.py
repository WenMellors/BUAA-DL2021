"""
Author:
    18377290 孙亦琦
    18231174 任杰瑞
"""

from trafficdl.model.abstract_model import AbstractModel

import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torch
from collections import defaultdict
from math import radians, cos, sin, asin, sqrt
from collections import deque
from tqdm import tqdm


# python run_model.py --task traj_loc_pred --model LSTPM --dataset foursquare_tky
class LSTPM(AbstractModel):

    def __init__(self, config, data_feature):
        super(LSTPM, self).__init__(config, data_feature)
        # print(config.__dict__)
        self.device = config['device']
        self.hidden_size = config['hidden_size']  # hidden_units
        self.hidden_units = config['hidden_size']  # hidden_units
        self.emb_size = config['emb_size']
        self.window_size = config['window_size']
        # self.user_dropout = config['dropout']

        self.loc_size = data_feature['loc_size']  # 数据集中POI点的数目。 # n_items
        self.tim_size = data_feature['tim_size']  # 时间窗口的大小，单位是小时。
        self.uid_size = data_feature['uid_size']  # 数据集中用户的数目。    # n_users
        print(f"self.uid_size: {self.uid_size}")
        self.loc_pad = data_feature['loc_pad']  # 补全轨迹所用的POI填充值。
        self.tim_pad = data_feature['tim_pad']  # 补全轨迹所用的时间编码。
        self.poi_profile = data_feature['poi_profile'].coordinates.apply(lambda x: eval(x))

        self.item_emb = torch.nn.Embedding(self.loc_size, self.emb_size)
        self.emb_tim = nn.Embedding(48, 10)
        self.lstmcell = nn.LSTM(input_size=self.emb_size, hidden_size=self.hidden_size)
        self.lstmcell_history = nn.LSTM(input_size=self.emb_size, hidden_size=self.hidden_size)
        self.linear = nn.Linear(self.hidden_size * 2, self.loc_size)
        self.dropout = nn.Dropout(0.0)
        self.user_dropout = nn.Dropout(config['dropout'])
        self.time_checkin_set = defaultdict(set)
        self.dilated_rnn = nn.LSTMCell(input_size=self.emb_size,
                                       hidden_size=self.hidden_size)  # could be the same as self.lstmcell
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()
        self.queue_ind = {}
        self.user_current = {}
        # self.poi_distance_matrix = self.caculate_poi_distance()
        self.poi_distance_matrix = {}

    #         print("finish_init")
    # todo:
    #  子任务1：
    #  填写参数 poi_coordinate
    #  论文中是data['vid_lookup']为参数，这是一个字典，以地点的id为key，以经纬度数组为value
    #  框架里面在 data_feature['poi_profile']里面，是一个DataFrame
    #  可以把框架DataFrame翻译成字典形式，也可以改函数，使用DataFrame，也可以两者都改
    #  建议不要直接翻译成字典，运算速度可能比较慢

    # poi_coordinate = data['vid_lookup']  # POI的坐标

    def caculate_poi_distance(self, poi_coors):
        # print("distance matrix")
        sim_matrix = np.zeros((len(poi_coors) + 1, len(poi_coors) + 1))
        for i in range(len(poi_coors)):
            for j in range(i, len(poi_coors)):
                poi_current = i + 1
                poi_target = j + 1
                poi_current_coor = poi_coors[poi_current]
                poi_target_coor = poi_coors[poi_target]
                distance_between = self.geodistance(poi_current_coor[1], poi_current_coor[0], poi_target_coor[1],
                                                    poi_target_coor[0])
                if distance_between < 1:
                    distance_between = 1
                sim_matrix[poi_current][poi_target] = distance_between
                sim_matrix[poi_target][poi_current] = distance_between
        # pickle.dump(sim_matrix, open('distance.pkl', 'wb'))
        return sim_matrix

    def get_poi_distance(self, poi_ids):
        for poi_id in poi_ids:
            if poi_id not in self.poi_distance_matrix:
                poi_value = self.poi_profile[poi_id]
                poi_line = self.poi_profile.apply(lambda x: self.geodistance(x[0], x[1], poi_value[0], poi_value[1]))
                self.poi_distance_matrix[poi_id] = poi_line.apply(lambda x: 1 if x < 1 else x)
        return np.array(list(self.poi_distance_matrix[poi_id] for poi_id in poi_ids))

    def geodistance(self, lng1, lat1, lng2, lat2):
        lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
        dlon = lng2 - lng1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        distance = 2 * asin(sqrt(a)) * 6371 * 1000
        distance = round(distance / 1000, 3)
        return distance

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def generate_input_history(self, batch):
        data_train = {}
        train_idx = {}
        for u in range(len(batch['uid'])):
            u_id = batch['uid'][u]
            train_id = [i for i in range(len(batch['history_loc'][u]) + 1)]
            train_idx[u_id] = train_id
            sessions_loc = batch['history_loc'][u]
            sessions_tim = batch['history_tim'][u]
            sessions_loc.append(batch['current_loc'][u])
            sessions_tim.append(batch['current_tim'][u])
            data_train[u_id] = {}

            for c, i in enumerate(train_id):
                if c == 0:
                    continue
                session_loc = sessions_loc[i]
                session_tim = sessions_tim[i]
                trace = {}
                loc_np = np.reshape(np.array([s.tolist() for s in session_loc[:-1]]),
                                    (len(session_loc[:-1]), 1))  # session去掉最后一个, POI
                tim_np = np.reshape(np.array([s.tolist() for s in session_tim[:-1]]),
                                    (len(session_tim[:-1]), 1))  # 将行向量变为列向量, time
                target = np.array([s.tolist() for s in session_loc[1:]])  # session去掉第一个, POI
                trace['loc'] = torch.LongTensor(loc_np.astype(int))
                trace['target'] = torch.LongTensor(target.astype(int))
                trace['tim'] = torch.LongTensor(tim_np.astype(int))

                history = []
                for j in range(c):
                    for k in range(len(sessions_loc[train_id[j]])):
                        history.append([sessions_loc[train_id[j]][k], sessions_tim[train_id[j]][k]])

                history = sorted(history, key=lambda x: x[1], reverse=False)  # 把history按时间升序排序
                history_loc = np.reshape(np.array([s[0].tolist() for s in history]), (len(history), 1))  # 历史地点，列向量
                history_tim = np.reshape(np.array([s[1].tolist() for s in history]), (len(history), 1))  # 历史时间，列向量
                trace['history_loc'] = torch.LongTensor(history_loc.astype(int))
                trace['history_tim'] = torch.LongTensor(history_tim.astype(int))

                data_train[u_id][i] = trace
        return data_train, train_idx

    def generate_queue(self, train_idx):
        user = list(train_idx.keys())
        train_queue = list()
        queue_ind = {}
        # model == 'random', mode2 == 'train'
        initial_queue = {}
        for u in user:
            initial_queue[u] = deque(train_idx[u][1:])
        queue_left = 1
        ind = 0
        while queue_left > 0:
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
                    u = u.item()
                    # print(f"ind:{ind}, u:{u},    :{train_queue[-1][1]}")
                    if self.user_current[u] == train_queue[-1][1] + 1:
                        queue_ind[u] = ind
                    ind += 1
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
        # print(queue_ind)
        self.queue_ind = queue_ind
        return train_queue

    def create_dilated_rnn_input(self, session_sequence_current):
        session_sequence_current = session_sequence_current.tolist()
        sequence_length = len(session_sequence_current)
        session_sequence_current.reverse()  # 把地点列表翻转       # 为什么要翻转啊，这里的翻转完全是用来处理逻辑的，但感觉翻转后逻辑还是好乱
        session_dilated_rnn_input_index = [0] * sequence_length  # 长度为sequence_length的零矩阵
        for i in range(sequence_length - 1):  # 不包括最后一个（即第一个地点）
            current_poi = [session_sequence_current[i]]  # 当前的地点
            poi_before = session_sequence_current[i + 1:]  # 之前的地点
            distance_row = self.get_poi_distance(current_poi)  # [[a,b,c,...,n]]
            distance_row_explicit = distance_row[:, poi_before][0]  # 把poi_before挑出来， 然后把[[...]]取出[...]
            index_closet = np.argmin(distance_row_explicit)  # 给出水平方向最小值的下标
            session_dilated_rnn_input_index[sequence_length - i - 1] = sequence_length - 2 - index_closet - i  # 没看懂
        session_sequence_current.reverse()  # 把地点列表翻转过来（python传列表会改变列表的值，所以要变回去）
        return session_dilated_rnn_input_index

    def generate_detailed_batch_data(self, batch, one_train_batch):
        session_id_batch = []  # 该batch中的session_id
        user_id_batch = []  # 该batch中的用户
        sequence_batch = []  # batch中所有session的所有地点[[p1,p2,..],[p1,...],...]
        sequences_lens_batch = []  # batch中每个session的POI数
        sequences_tim_batch = []  # batch中所有session的所有时间[[t1,t2,..],[t1,...],...]
        sequences_dilated_input_batch = []  # batch中所有session的修改后index[[idx1,idx2,..],[idx1,...],...]
        for sample in tqdm(one_train_batch):  # 遍历每个对应session
            # print(sample)
            user_id_batch.append(sample[0].tolist())
            session_id_batch.append(sample[1])
            idx = 0
            for i in range(len(batch['uid'])):
                if batch['uid'][i] == sample[0]:
                    idx = i
                    break
            if len(batch['history_loc'][idx]) <= sample[1]:
                print(f"user:{idx}, ind:{sample[1]}")
                session_sequence_current = batch['current_loc'][idx]
                session_sequence_tim_current = batch['current_tim'][idx]
            else:
                session_sequence_current = batch['history_loc'][idx][sample[1]]
                session_sequence_tim_current = batch['history_tim'][idx][sample[1]]  # session中的时间列表

            # todo:
            #  子任务2：这里的data_neural要用batch解析出来
            session_sequence_dilated_input = self.create_dilated_rnn_input(session_sequence_current)
            # 把距离矩阵传了进去（明明这里也是从main里面获取的，是传了个寂寞吗）
            sequence_batch.append(session_sequence_current.tolist())
            sequences_lens_batch.append(len(session_sequence_current))
            sequences_tim_batch.append(session_sequence_tim_current)
            sequences_dilated_input_batch.append(session_sequence_dilated_input)
        return user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, \
            sequences_tim_batch, sequences_dilated_input_batch

    # 数据补齐，把每组数据按最大数据补齐（用0补齐）
    def pad_batch_of_lists_masks(self, batch_of_lists, max_len):
        padded = [length + [0] * (max_len - len(length)) for length in batch_of_lists]  # 把未对齐数据用0补齐
        padded_mask = [[1.0] * (len(length) - 1) + [0.0] * (max_len - len(length) + 1) for length in batch_of_lists]
        padde_mask_non_local = [[1.0] * (len(length)) + [0.0] * (max_len - len(length)) for length in
                                batch_of_lists]  # 补齐标志， 真实数据标记为1， 补齐所用数据标记为0
        return padded, padded_mask, padde_mask_non_local

    def caculate_time_sim(self, batch):
        for u in range(len(batch['history_loc'])):
            for sid in range(len(batch['history_loc'][u])):
                for checkin in range(len(batch['history_loc'][u][sid])):
                    loc = batch['history_loc'][u][sid][checkin]
                    tim = batch['history_tim'][u][sid][checkin]
                    if tim not in self.time_checkin_set:
                        self.time_checkin_set[tim] = set()
                    self.time_checkin_set[tim].add(loc)
        sim_matrix = np.zeros((self.window_size + 1, self.window_size + 1))
        for i in range(self.window_size + 1):
            for j in range(self.window_size + 1):
                set_i = self.time_checkin_set[i]
                set_j = self.time_checkin_set[j]
                set_or = len(set_i | set_j)
                if set_or != 0:
                    jaccard_ij = len(set_i & set_j) / set_or
                    sim_matrix[i][j] = jaccard_ij
        return sim_matrix

    def forward(self, batch):
        torch.cuda.empty_cache()
        # print("begin forward!")
        # print(f"history_loc size : {batch['uid'][0]}: {len(batch['history_loc'][0])}")
        # print(f"history_loc size : {batch['uid'][1]}: {len(batch['history_loc'][1])}")
        tim_sim_matrix = self.caculate_time_sim(batch)
        # print("finish caculate_time_sim")

        data_train, train_idx = self.generate_input_history(batch)

        self.user_current = {}
        for u in range(len(batch['uid'])):
            self.user_current[batch['uid'][u].item()] = len(batch['history_loc'][u])
        # print(self.user_current)
        # print()

        # print("finish generate_input_history")
        one_train_batch = self.generate_queue(train_idx)
        # print("finish generate_queue")
        # user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, sequence_tim_batch,
        # sequence_dilated_rnn_index_batch = \
        #     self.generate_detailed_batch_data(batch=batch, one_train_batch=one_train_batch)
        user_vectors, session_id_batch, sequence_batch, sequences_lens_batch, \
            sequence_tim_batch, sequence_dilated_rnn_index_batch = \
            self.generate_detailed_batch_data(batch=batch, one_train_batch=one_train_batch)
        # print("finish generate_detailed_batch_data")
        max_len = max(sequences_lens_batch)  # 每个session的POI数 的最大值
        padded_sequence_batch, mask_batch_ix, mask_batch_ix_non_local = \
            self.pad_batch_of_lists_masks(sequence_batch, max_len)  # 用0补齐数据
        item_vectors = torch.LongTensor(np.array(padded_sequence_batch).astype(int)).to(self.device)
        mask_batch_ix_non_local = torch.FloatTensor(np.array(mask_batch_ix_non_local).astype(float)).to(self.device)
        is_train = True

        # todo: 以下是原论文的forward部分，以上是通过batch获取forward所需参数
        # todo: 需要研究一下 is_train 是干什么的，会产生哪些影响
        # print("begin forward")
        batch_size = item_vectors.size()[0]
        # print("batch_size : ", batch_size)
        sequence_size = item_vectors.size()[1]
        items = self.item_emb(item_vectors)
        item_vectors = item_vectors.tolist()
        x = items
        x = x.transpose(0, 1)
        h1 = torch.zeros(1, batch_size, self.hidden_units).to(self.device)
        c1 = torch.zeros(1, batch_size, self.hidden_units).to(self.device)
        out, (h1, c1) = self.lstmcell(x, (h1, c1))
        del h1, c1
        out = out.transpose(0, 1)  # batch_size * sequence_length * embedding_dim
        x1 = items
        # ###########################################################
        user_batch = np.array(user_vectors)
        y_list = []
        out_hie = []
        for ii in range(batch_size):
            ##########################################
            current_session_input_dilated_rnn_index = sequence_dilated_rnn_index_batch[ii]
            hiddens_current = x1[ii]
            dilated_lstm_outs_h = []
            dilated_lstm_outs_c = []
            for index_dilated in range(len(current_session_input_dilated_rnn_index)):
                index_dilated_explicit = current_session_input_dilated_rnn_index[index_dilated]
                hidden_current = hiddens_current[index_dilated].unsqueeze(0)
                if index_dilated == 0:
                    h = torch.zeros(1, self.hidden_units).to(self.device)
                    c = torch.zeros(1, self.hidden_units).to(self.device)
                    (h, c) = self.dilated_rnn(hidden_current, (h, c))
                    dilated_lstm_outs_h.append(h)
                    dilated_lstm_outs_c.append(c)
                else:
                    (h, c) = self.dilated_rnn(hidden_current, (
                        dilated_lstm_outs_h[index_dilated_explicit], dilated_lstm_outs_c[index_dilated_explicit]))
                    dilated_lstm_outs_h.append(h)
                    dilated_lstm_outs_c.append(c)
            dilated_lstm_outs_h.append(hiddens_current[len(current_session_input_dilated_rnn_index):])
            dilated_out = torch.cat(dilated_lstm_outs_h, dim=0).unsqueeze(0)
            out_hie.append(dilated_out)
            user_id_current = user_batch[ii]
            current_session_timid = sequence_tim_batch[ii][:-1]
            current_session_poiid = item_vectors[ii][:len(current_session_timid)]
            session_id_current = session_id_batch[ii]
            current_session_embed = out[ii]
            current_session_mask = mask_batch_ix_non_local[ii].unsqueeze(1)
            sequence_length = int(sum(np.array(current_session_mask.cpu()))[0])
            current_session_represent_list = []
            if is_train:
                for iii in range(sequence_length - 1):
                    current_session_represent = torch.sum(current_session_embed * current_session_mask,
                                                          dim=0).unsqueeze(0) / sum(current_session_mask)
                    current_session_represent_list.append(current_session_represent)
            else:
                for iii in range(sequence_length - 1):
                    current_session_represent_rep_item = current_session_embed[0:iii + 1]
                    current_session_represent_rep_item = torch.sum(current_session_represent_rep_item, dim=0).unsqueeze(
                        0) / (iii + 1)
                    current_session_represent_list.append(current_session_represent_rep_item)

            current_session_represent = torch.cat(current_session_represent_list, dim=0)
            list_for_sessions = []
            list_for_avg_distance = []
            h2 = torch.zeros(1, 1, self.hidden_units).to(self.device)  # whole sequence
            c2 = torch.zeros(1, 1, self.hidden_units).to(self.device)
            for jj in range(session_id_current):
                # sequence = [s[0] for s in self.data_neural[user_id_current]['sessions'][jj]]
                idx_inbatch = 0
                for user_batch_id in range(len(batch['uid'])):
                    if batch['uid'][user_batch_id] == user_id_current:
                        idx_inbatch = user_batch_id
                        break
                if len(batch['history_loc'][idx_inbatch]) <= jj:
                    sequence = batch['current_loc'][idx_inbatch]
                else:
                    sequence = batch['history_loc'][idx_inbatch][jj]
                sequence_emb = self.item_emb(sequence).unsqueeze(1)
                sequence = sequence.cpu()
                sequence_emb, (h2, c2) = self.lstmcell_history(sequence_emb, (h2, c2))
                # sequence_tim_id = [s[1] for s in self.data_neural[user_id_current]['sessions'][jj]]
                if len(batch['history_tim'][idx_inbatch]) <= jj:
                    sequence_tim_id = batch['current_tim'][idx_inbatch].tolist()
                else:
                    sequence_tim_id = batch['history_tim'][idx_inbatch][jj].tolist()

                # jaccard_sim_row = tim_sim_matrix[current_session_timid.cpu()]
                # print("current_session_timid", current_session_timid.cpu())
                jaccard_sim_row = torch.FloatTensor(tim_sim_matrix[current_session_timid.cpu()].astype(float)).to(
                    self.device)
                jaccard_sim_expicit = jaccard_sim_row[:, sequence_tim_id]
                distance_row = self.get_poi_distance(current_session_poiid)
                distance_row_expicit = torch.FloatTensor(distance_row[:, sequence.cpu()].astype(float)).to(self.device)
                distance_row_expicit_avg = torch.mean(distance_row_expicit, dim=1)
                jaccard_sim_expicit_last = f.softmax(jaccard_sim_expicit)
                hidden_sequence_for_current1 = torch.mm(jaccard_sim_expicit_last, sequence_emb.squeeze(1))
                hidden_sequence_for_current = hidden_sequence_for_current1
                list_for_sessions.append(hidden_sequence_for_current.unsqueeze(0))
                list_for_avg_distance.append(distance_row_expicit_avg.unsqueeze(0))
            del h2, c2
            avg_distance = torch.cat(list_for_avg_distance, dim=0).transpose(0, 1)
            sessions_represent = torch.cat(list_for_sessions, dim=0).transpose(0, 1)
            # current_items * history_session_length * embedding_size
            current_session_represent = current_session_represent.unsqueeze(2)  # current_items * embedding_size * 1
            sims = f.softmax(sessions_represent.bmm(current_session_represent).squeeze(2), dim=1).unsqueeze(
                1)  # ==> current_items * 1 * history_session_length
            # out_y_current = sims.bmm(sessions_represent).squeeze(1)
            out_y_current = torch.selu(self.linear1(sims.bmm(sessions_represent).squeeze(1)))
            del sims
            # layer_2
            # layer_2_current = (lambda*out_y_current +
            # (1-lambda)*current_session_embed[:sequence_length-1]).unsqueeze(2)
            # #lambda from [0.1-0.9] better performance
            # layer_2_current = (out_y_current + current_session_embed[:sequence_length-1]).unsqueeze(2)
            # ==>current_items * embedding_size * 1
            layer_2_current = (0.5 * out_y_current + 0.5 * current_session_embed[:sequence_length - 1]).unsqueeze(2)

            del out_y_current
            layer_2_sims = f.softmax(sessions_represent.bmm(layer_2_current).squeeze(2) * 1.0 / avg_distance,
                                     dim=1).unsqueeze(1)  # ==>>current_items * 1 * history_session_length
            del layer_2_current
            out_layer_2 = layer_2_sims.bmm(sessions_represent).squeeze(1)
            del layer_2_sims
            del sessions_represent
            out_y_current_padd = torch.FloatTensor(sequence_size - sequence_length + 1, self.emb_size).zero_().to(
                self.device)
            out_layer_2_list = []
            out_layer_2_list.append(out_layer_2)
            out_layer_2_list.append(out_y_current_padd)
            out_layer_2 = torch.cat(out_layer_2_list, dim=0).unsqueeze(0)
            y_list.append(out_layer_2)
            # print(f"batch{ii} finish")
        y = torch.selu(torch.cat(y_list, dim=0))
        out_hie = f.selu(torch.cat(out_hie, dim=0))
        out = f.selu(out)

        out = (out + out_hie) * 0.5
        del out_hie

        out_put_emb_v1 = torch.cat([y, out], dim=2)
        output_ln = self.linear(out_put_emb_v1)
        del out_put_emb_v1
        output = f.log_softmax(output_ln, dim=-1)
        return output, mask_batch_ix, padded_sequence_batch

    def predict(self, batch):
        """
        参数说明:
            batch (trafficdl.data.batch): 类 dict 文件，其中包含的键值参见任务说明文件。
        返回值:
            score (pytorch.tensor): 对应张量 shape 应为 batch_size *
                loc_size。这里返回的是模型对于输入当前轨迹的下一跳位置的预测值。
        """
        # todo:
        #  子任务3：
        #  这里返回的格式不一定对，猜测也要模仿论文里面train_network里面的码
        logp_seq, mask_batch_ix, padded_sequence_batch = self.forward(batch)
        del padded_sequence_batch
        mask_batch_ix = torch.LongTensor(np.array(mask_batch_ix).astype(float)).to(self.device)
        predictions_logp = logp_seq[:, :-1] * mask_batch_ix[:, :-1, None]
        del mask_batch_ix

        # predictions_logp是一个三维张量，[x,y,z] x序列id, y序列（填充后), z POIs
        # 现在self.queue_ind 是一个字典，存有每个用户对应的测试数据的序列id(x), key是user_id
        output = []
        for i in range(len(batch['uid'])):
            uid = batch['uid'][i].item()
            queue_id = self.queue_ind[uid]
            next_idx = len(batch['current_loc'][i]) - 2
            output.append(predictions_logp[queue_id, next_idx, :].tolist())
        output = torch.LongTensor(np.array(output).astype(float)).to(self.device)
        return output

    def calculate_loss(self, batch):
        """
        参数说明:
            batch (trafficdl.data.batch): 类 dict 文件，其中包含的键值参见任务说明文件。
        返回值:
            loss (pytorch.tensor): 可以调用 pytorch 实现的 loss 函数与 batch['target']
                目标值进行 loss 计算，并将计算结果返回。如模型有自己独特的 loss 计算方式则自行参考实现。
        """
        # todo:
        #  子任务4:
        #  这里没写，猜测也要模仿论文里面train_network里面的码
        # criterion = nn.NLLLoss().to(self.device)
        # print("in calculate_loss")
        logp_seq, mask_batch_ix, padded_sequence_batch = self.forward(batch)
        mask_batch_ix = torch.LongTensor(np.array(mask_batch_ix).astype(float)).to(self.device)
        padded_sequence_batch = torch.LongTensor(np.array(padded_sequence_batch).astype(float)).to(self.device)
        predictions_logp = logp_seq[:, :-1] * mask_batch_ix[:, :-1, None]
        actual_next_tokens = padded_sequence_batch[:, 1:]
        logp_next = torch.gather(predictions_logp, dim=2, index=actual_next_tokens[:, :, None])
        loss = -logp_next.sum() / mask_batch_ix[:, :-1].sum()
        del logp_seq
        del mask_batch_ix
        del padded_sequence_batch
        del predictions_logp
        del actual_next_tokens
        del logp_next
        return loss
