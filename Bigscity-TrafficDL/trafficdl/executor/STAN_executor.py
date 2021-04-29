import json
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import time
from trafficdl.model import STAN
import random

from trafficdl.executor.abstract_executor import AbstractExecutor
from trafficdl.utils import get_evaluator

part = 100
max_len = 100


def to_npy(x, device):
    return x.cpu().data.numpy() if device == 'cuda' else x.cpu().detach().numpy()


def calculate_acc(prob, label, device):
    # log_prob (N, L), label (N), batch_size [*M]
    acc_train = [0, 0, 0, 0]
    for i, k in enumerate([1, 5, 10, 20]):
        # topk_batch (N, k)
        _, topk_predict_batch = torch.topk(prob, k=k)
        for j, topk_predict in enumerate(to_npy(topk_predict_batch, device)):
            # topk_predict (k)
            if to_npy(label, device)[j] in topk_predict:
                acc_train[i] += 1

    return np.array(acc_train)


def sampling_prob(prob, label, num_neg):
    num_label, l_m = prob.shape[0], prob.shape[1]-1  # prob (N, L)
    label = label.view(-1)  # label (N)
    init_label = np.linspace(0, num_label-1, num_label)  # (N), [0 -- num_label-1]
    init_prob = torch.zeros(size=(num_label, num_neg+len(label)))  # (N, num_neg+num_label)

    random_ig = random.sample(range(1, l_m+1), num_neg)  # (num_neg) from (1 -- l_max)
    while len([lab for lab in label if lab in random_ig]) != 0:  # no intersection
        random_ig = random.sample(range(1, l_m+1), num_neg)

    # global global_seed
    # random.seed(global_seed)
    # global_seed += 1

    # place the pos labels ahead and neg samples in the end
    for k in range(num_label):
        for i in range(num_neg + len(label)):
            if i < len(label):
                init_prob[k, i] = prob[k, label[i]]
            else:
                init_prob[k, i] = prob[k, random_ig[i-len(label)]]

    return torch.FloatTensor(init_prob), torch.LongTensor(init_label)  # (N, num_neg+num_label), (N)


class StanTrajLocPredExecutor(AbstractExecutor):

    def __init__(self, config, model):
        self.config = config
        # load other parameters
        self.device = self.config['device']
        self.model = model.to(self.device)
        self.start = time.time()
        self.records = {'epoch': [], 'acc_valid': [], 'acc_test': []}
        self.start_epoch = 1
        self.num_neg = 10
        self.interval = 1000
        self.batch_size = 4 # N = 1
        self.learning_rate = 3e-3
        self.num_epoch = self.config['max_epoch']
        self.threshold = 0  # 0 if not update

        self.evaluator = get_evaluator(config)
        self.metrics = 'Recall@{}'.format(config['topk'])
        self.model = model.to(self.config['device'])
        self.tmp_path = './trafficdl/tmp/checkpoint/'
        self.cache_dir = './trafficdl/cache/model_cache'
        self.evaluate_res_dir = './trafficdl/cache/evaluate_cache'
        self.loss_func = None  # TODO: 根据配置文件支持选择特定的 Loss Func 目前并未实装

    def train(self, train_dataloader, eval_dataloader):
        # set optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)

        for t in range(self.num_epoch):
            # settings or validation and test
            valid_size, test_size = 0, 0
            acc_valid, acc_test = [0, 0, 0, 0], [0, 0, 0, 0]

            bar = tqdm(total=part)
            for step, item in enumerate(train_dataloader):
                # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
                person_input, person_m1, person_m2t, person_label, person_traj_len = item

                # first, try batch_size = 1 and mini_batch = 1

                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long).to(self.device)
                m1_mask = torch.zeros((self.batch_size, max_len, max_len, 2), dtype=torch.float32).to(self.device)
                for mask_len in range(1, person_traj_len[0]+1):  # from 1 -> len
                    # if mask_len != person_traj_len[0]:
                    #     continue
                    input_mask[:, :mask_len] = 1.
                    m1_mask[:, :mask_len, :mask_len] = 1.

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    train_label = person_label[:, mask_len - 1]  # (N)
                    train_len = torch.zeros(size=(self.batch_size,), dtype=torch.long).to(self.device) + mask_len

                    prob = self.model(train_input, train_m1, train_m2t, train_len)  # (N, L)

                    if mask_len <= person_traj_len[0] - 2:  # only training
                        # nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                        prob_sample, label_sample = sampling_prob(prob, train_label, self.num_neg)
                        loss_train = F.cross_entropy(prob_sample, label_sample)
                        loss_train.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    elif mask_len == person_traj_len[0] - 1:  # only validation
                        valid_size += person_input.shape[0]
                        # v_prob_sample, v_label_sample = sampling_prob(prob_valid, valid_label, self.num_neg)
                        # loss_valid += F.cross_entropy(v_prob_sample, v_label_sample, reduction='sum')
                        acc_valid += calculate_acc(prob, train_label, self.device)

                    elif mask_len == person_traj_len[0]:  # only test
                        test_size += person_input.shape[0]
                        # v_prob_sample, v_label_sample = sampling_prob(prob_valid, valid_label, self.num_neg)
                        # loss_valid += F.cross_entropy(v_prob_sample, v_label_sample, reduction='sum')
                        acc_test += calculate_acc(prob, train_label, self.device)

                bar.update(self.batch_size)
            bar.close()

            acc_valid = np.array(acc_valid) / valid_size
            print('epoch:{}, time:{}, valid_acc:{}'.format(self.start_epoch + t, time.time() - self.start, acc_valid))

            acc_test = np.array(acc_test) / test_size
            print('epoch:{}, time:{}, test_acc:{}'.format(self.start_epoch + t, time.time() - self.start, acc_test))

            self.records['acc_valid'].append(acc_valid)
            self.records['acc_test'].append(acc_test)
            self.records['epoch'].append(self.start_epoch + t)

            if self.threshold < np.mean(acc_valid):
                self.threshold = np.mean(acc_valid)
                # save the model
                model_cache_file = \
                    './trafficdl/cache/model_cache/{}_{}.m'.format(
                        self.config['model'], self.config['dataset'])
                self.save_model(model_cache_file)

    def evaluate(self, test_dataloader):
        user_ids = []
        for t in range(self.num_epoch):
            # settings or validation and test
            valid_size, test_size = 0, 0
            acc_valid, acc_test = [0, 0, 0, 0], [0, 0, 0, 0]
            cum_valid, cum_test = [0, 0, 0, 0], [0, 0, 0, 0]

            for step, item in enumerate(test_dataloader):
                # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
                person_input, person_m1, person_m2t, person_label, person_traj_len = item

                # first, try batch_size = 1 and mini_batch = 1

                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long).to(self.device)
                m1_mask = torch.zeros((self.batch_size, max_len, max_len, 2), dtype=torch.float32).to(self.device)
                for mask_len in range(1, person_traj_len[0] + 1):  # from 1 -> len
                    # if mask_len != person_traj_len[0]:
                    #     continue
                    input_mask[:, :mask_len] = 1.
                    m1_mask[:, :mask_len, :mask_len] = 1.

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    train_label = person_label[:, mask_len - 1]  # (N)
                    train_len = torch.zeros(size=(self.batch_size,), dtype=torch.long).to(self.device) + mask_len

                    prob = self.model(train_input, train_m1, train_m2t, train_len)  # (N, L)

                    if mask_len <= person_traj_len[0] - 2:  # only training
                        continue

                    elif mask_len == person_traj_len[0] - 1:  # only validation
                        acc_valid = calculate_acc(prob, train_label, self.device)
                        cum_valid += calculate_acc(prob, train_label, self.device)

                    elif mask_len == person_traj_len[0]:  # only test
                        acc_test = calculate_acc(prob, train_label, self.device)
                        cum_test += calculate_acc(prob, train_label, self.device)

                print(step, acc_valid, acc_test)

                if acc_valid.sum() == 0 and acc_test.sum() == 0:
                    user_ids.append(step)

    def _valid_epoch(self, data_loader, model, total_batch, verbose):
        model.train(False)
        self.evaluator.clear()
        cnt = 0
        for batch in data_loader:
            batch.to_tensor(device=self.config['device'])
            scores = model.predict(batch)
            evaluate_input = {
                'uid': batch['uid'].tolist(),
                'loc_true': batch['target'].tolist(),
                'loc_pred': scores.tolist()
            }
            cnt += 1
            if cnt % verbose == 0:
                print('finish batch {}/{}'.format(cnt, total_batch))
            self.evaluator.collect(evaluate_input)
        avg_acc = self.evaluator.evaluate()[self.metrics]  # 随便选一个就行
        return avg_acc
    
    def save_model(self, cache_name):
        torch.save({
            'state_dict': self.model.state_dict(),
            'records': self.records,
            'time': time.time() - self.start
        }, cache_name)
    
    def load_model(self, cache_name):
        self.model.load_state_dict(torch.load(cache_name))
