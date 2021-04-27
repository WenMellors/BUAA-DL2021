from trafficdl.model.abstract_model import AbstractModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum

class Rnn(Enum):
    ''' The available RNN units '''
    
    RNN = 0
    GRU = 1    
    LSTM = 2    
    
    @staticmethod
    def from_string(name):
        if name == 'rnn' or name == 'RNN':
            return Rnn.RNN
        if name == 'gru' or name == 'GRU':
            return Rnn.GRU
        if name == 'lstm' or name == 'LSTM':
            return Rnn.LSTM        
        raise ValueError('{} not supported in --rnn'.format(name))        

class RnnFactory():
    ''' Creates the desired RNN unit. '''
    
    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)
                
    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return 'Use pytorch RNN implementation.'
        if self.rnn_type == Rnn.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == Rnn.LSTM:
            return 'Use pytorch LSTM implementation.'        
    
    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]
        
    def create(self, hidden_size):
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)

class Flashback(nn.Module):
    ''' Flashback RNN: Applies weighted average using spatial and tempoarl data in combination
    of user embeddings to the output of a generic RNN unit (RNN, GRU, LSTM).
    '''    
    
    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, rnn_factory):
        super().__init__()
        self.input_size = input_size
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.f_t = f_t # function for computing temporal weight
        self.f_s = f_s # function for computing spatial weight

        self.encoder = nn.Embedding(input_size, hidden_size) # location embedding
        self.user_encoder = nn.Embedding(user_count, hidden_size) # user embedding
        self.rnn = rnn_factory.create(hidden_size)
        self.fc = nn.Linear(2*hidden_size, input_size) # create outputs in lenght of locations

    def forward(self, x, t, s, y_t, y_s, h, active_user):        
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)        
        out, h = self.rnn(x_emb, h)
        
        # comopute weights per user
        out_w = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)
        for i in range(seq_len):
            sum_w = torch.zeros(user_len, 1, device=x.device)
            for j in range(i+1):
                dist_t = t[i] - t[j]
                dist_s = torch.norm(s[i] - s[j], dim=-1)
                a_j = self.f_t(dist_t, user_len)
                b_j = self.f_s(dist_s, user_len)
                a_j = a_j.unsqueeze(1)
                b_j = b_j.unsqueeze(1)
                w_j = a_j*b_j + 1e-10 # small epsilon to avoid 0 division
                sum_w += w_j
                out_w[i] += w_j*out[j]
            # normalize according to weights
            out_w[i] /= sum_w
        
        # add user embedding:
        p_u = self.user_encoder(active_user)
        p_u = p_u.view(user_len, self.hidden_size)
        out_pu = torch.zeros(seq_len, user_len, 2*self.hidden_size, device=x.device)
        for i in range(seq_len):
            out_pu[i] = torch.cat([out_w[i], p_u], dim=1)
        y_linear = self.fc(out_pu)
        return y_linear, h

'''
~~~ h_0 strategies ~~~
Initialize RNNs hidden states
'''

def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:        
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))        
    else:        
        return FixNoiseStrategy(hidden_size)

class H0Strategy():
    
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
    
    def on_init(self, user_len, device):
        pass
    
    def on_reset(self, user):
        pass
    
    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    ''' use fixed normal noise as initialization '''
    
    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1/self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu
    
    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)
    
    def on_reset(self, user):
        return self.h0

class LstmStrategy(H0Strategy):
    ''' creates h0 and c0 using the inner strategy '''
    
    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy
    
    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return (h,c)
    
    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return (h,c)
        
class FlashbackTrainer(AbstractModel):
    ''' Instantiates Flashback module with spatial and temporal weight functions.
    Performs loss computation and prediction.
    '''
    
    def __init__(self, config, data_feature, lambda_t=0.1, lambda_s=1000):
        ''' The hyper parameters to control spatial and temporal decay.
        '''
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s

        super(FlashbackTrainer, self).__init__(config, data_feature)
        self.loc_size = data_feature['loc_size']
        self.uid_size = data_feature['uid_size']
        self.hidden_size = config['hidden_size']
        self.device = config['device']
        self.rnn_type = config['rnn_type']
        
        self.prepare(RnnFactory(self.rnn_type))

        self.loc_emb_size = config['loc_emb_size']
        self.tim_size = data_feature['tim_size']
        self.tim_emb_size = config['tim_emb_size']

        self.emb_loc = nn.Embedding(
            self.loc_size, self.loc_emb_size,
            padding_idx=data_feature['loc_pad'])
        self.emb_tim = nn.Embedding(
            self.tim_size, self.tim_emb_size,
            padding_idx=data_feature['tim_pad'])
    
    def __str__(self):
        return 'Use flashback training.'        
    
    def parameters(self):   
        return self.model.parameters()
    
    def prepare(self, gru_factory):
        # loc_count -> number of locations in dataset (self.loc_size)
        # user_count -> number of users in dataset (self.uid_size)
        # hidden_size -> self.hidden_size (config['hidden_size'])
        # gru_factory -> RnnFactory(self.rnn_type)
        # device -> self.device
        f_t = lambda delta_t, user_len: ((torch.cos(delta_t*2*np.pi/86400) + 1) / 2)*torch.exp(-(delta_t/86400*self.lambda_t)) # hover cosine + exp decay
        f_s = lambda delta_s, user_len: torch.exp(-(delta_s*self.lambda_s)) # exp decay   
        self.loc_count = self.loc_size
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.model = Flashback(self.loc_size, self.uid_size, self.hidden_size, f_t, f_s, gru_factory).to(self.device)
    
    def evaluate(self, x, t, s, y_t, y_s, h, active_users):
        ''' takes a batch (users x location sequence)
        then does the prediction and returns a list of user x sequence x location
        describing the probabilities for each location at each position in the sequence.
        t, s are temporal and spatial data related to the location sequence x
        y_t, y_s are temporal and spatial data related to the target sequence y.
        Flashback does not access y_t and y_s for prediction!
        '''
        # x -> (20, 200)
        # t -> (20, 200)
        # s -> (20, 200, 2)
        # y -> (20, 200)
        # y_t -> (20, 200)
        # y_s -> (20, 200, 2)
        # h -> (1, 200, 20)
        # active_users -> (1, 200)
        
        self.model.eval()
        out, h = self.forward(x, t, s, y_t, y_s, h, active_users)
        out_t = out.transpose(0, 1)
        return out_t, h # model outputs logits
    
    def predict(self, batch):
        """
        参数说明:
            batch (trafficdl.data.batch): 类 dict 文件，其中包含的键值参见任务说明文件。
        返回值:
            score (pytorch.tensor): 对应张量 shape 应为 batch_size *
                loc_size。这里返回的是模型对于输入当前轨迹的下一跳位置的预测值。
        """
        x = batch['uid'] # not sure ...
        t = batch['history_tim']
        s = batch['history_loc']
        y_t = batch['target_tim']
        y_s = batch['target_loc']

        if self.rnn_type == 'lstm' or self.rnn_type == 'LSTM':
            self.is_lstm = True
        else:
            self.is_lstm = False

        h0_strategy = create_h0_strategy(self.hidden_size, self.is_lstm)
        h = h0_strategy.on_init(self.batch_size, self.device)
        active_users = batch['uid']

        return self.evaluate(x, t, s, y_t, y_s, h, active_users)

    def calculate_loss(self, batch):
        ''' takes a batch (users x location sequence)
        and corresponding targets in order to compute the training loss '''
        
        criterion = nn.NLLLoss().to(self.device)
        scores = self.forward(batch)
        return criterion(scores, batch['target'])
        



class TemplateTLP(AbstractModel):
    """
    请参考开源模型代码，完成本文件的编写。请务必重写 __init__, predict, calculate_loss 三个方法。
    """

    def __init__(self, config, data_feature):
        """
        参数说明：
            config (dict): 配置模块根据模型对应的 config.json 文件与命令行传递的参数
                根据 config 初始化模型参数
            data_feature (dict): 在数据预处理步骤提取到的数据集所属的特征参数，如 loc_size，uid_size 等。
        """

    def predict(self, batch):
        """
        参数说明:
            batch (trafficdl.data.batch): 类 dict 文件，其中包含的键值参见任务说明文件。
        返回值:
            score (pytorch.tensor): 对应张量 shape 应为 batch_size *
                loc_size。这里返回的是模型对于输入当前轨迹的下一跳位置的预测值。
        """

    def calculate_loss(self, batch):
        """
        参数说明:
            batch (trafficdl.data.batch): 类 dict 文件，其中包含的键值参见任务说明文件。
        返回值:
            loss (pytorch.tensor): 可以调用 pytorch 实现的 loss 函数与 batch['target']
                目标值进行 loss 计算，并将计算结果返回。如模型有自己独特的 loss 计算方式则自行参考实现。
        """
