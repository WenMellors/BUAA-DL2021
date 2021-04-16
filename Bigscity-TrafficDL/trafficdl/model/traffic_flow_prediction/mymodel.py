import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from logging import getLogger
import torch
from trafficdl.model import loss
from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel

class MLP(nn.Module):
    def __init__(self, ninp, nhid, nout, nlayers, dropout):
        super(MLP, self).__init__()
        self.ninp = ninp
        # modules
        if nlayers == 1:
            self.module = nn.Linear(ninp, nout)
        else:
            modules = [nn.Linear(ninp, nhid), nn.ReLU(), nn.Dropout(dropout)]
            nlayers -= 1
            while nlayers > 1:
                modules += [nn.Linear(nhid, nhid), nn.ReLU(), nn.Dropout(dropout)]
                nlayers -= 1
            modules.append(nn.Linear(nhid, nout))
            self.module = nn.Sequential(*modules)

    def forward(self, input):
        return self.module(input)


class mymodel(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        """
        构造模型
        :param config: 源于各种配置的配置字典
        :param data_feature: 从数据集Dataset类的`get_data_feature()`接口返回的必要的数据相关的特征
        """
        # 1.初始化父类（必须）
        super().__init__(config, data_feature)
        # 2.从data_feature获取想要的信息，注意不同模型使用不同的Dataset类，其返回的data_feature内容不同（必须）
        # 以TrafficStateGridDataset为例演示取数据，可以取出如下的数据，用不到的可以不取
        # **这些参数的不能从config中取的**
        self._scaler = self.data_feature.get('scaler')  # 用于数据归一化
        self.adj_mx = self.data_feature.get('adj_mx', 1)  # 邻接矩阵
        self.num_nodes = self.data_feature.get('num_nodes', 1)  # 网格个数
        self.feature_dim = self.data_feature.get('feature_dim', 1)  # 输入维度
        self.output_dim = self.data_feature.get('output_dim', 1)  # 输出维度
        #self.len_row = self.data_feature.get('len_row', 1)  # 网格行数
        #self.len_column = self.data_feature.get('len_column', 1)  # 网格列数
        # 3.初始化log用于必要的输出（必须）
        self._logger = getLogger()
        # 4.初始化device（必须）
        self.device = config.get('device', torch.device('cpu'))
        # 5.初始化输入输出时间步的长度（非必须）
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        # 6.从config中取用到的其他参数，主要是用于构造模型结构的参数（必须）
        # 这些涉及到模型结构的参数应该放在trafficdl/config/model/model_name.json中（必须）
        # 例如: self.blocks = config['blocks']
        # ...
        # 7.构造深度模型的层次结构（必须）
        # 例如: 使用简单RNN: self.rnn = nn.GRU(input_size, hidden_size, num_layers)
        #get from config
        mode = config['mode']
        nhid = config["nhid"]
        nlayers = config["nlayers"]
        dropout_f =config["dropout_f"]
        dropout_d = config["dropout_d"]
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        
        #copy from stnn
        self.nt = self.input_window
        self.nx = self.num_nodes
        self.nz = self.feature_dim
        self.mode = mode
        
        relations = torch.Tensor(self.adj_mx).unsqueeze(1)
        # kernel
        self.activation = F.tanh 
        device = self.device
        if mode is None or mode == 'refine':
            self.relations = torch.cat((torch.eye(self.nx).unsqueeze(1), relations), 1)
        elif mode == 'discover':
            self.relations = torch.cat((torch.eye(self.nx).unsqueeze(1),
                                        torch.ones(self.nx, 1, self.nx).to(device)), 1)
        self.nr = self.relations.size(1)
        # modules
        self.drop = nn.Dropout(dropout_f)
        self.factors = nn.Parameter(torch.Tensor(self.nt, self.nx, self.nz))
        self.sigmo = nn.Sigmoid()
        self.ffunc = nn.Linear(self.nt*self.nx,2 * self.nt*self.nx)
        #1 self.factf = nn.Parameter(torch.eye(self.nt)) # time feature
        #1 self.facts = nn.Parameter(torch.eye(self.nx)) # loc feature
        #self.factfunc = nn.Linear(self.nt*self.nx,2*self.nt*self.nx) # val feature
        self.dynamic = MLP(self.nz * self.nr, nhid, self.nz, nlayers, dropout_d)
        self.decoder = nn.Linear(self.nz, 1, bias=False)
        if mode == 'refine':
            self.relations.data = self.relations.data.ceil().clamp(0, 1).byte()
            self.rel_weights = nn.Parameter(torch.Tensor(self.relations.sum().item() - self.nx))
        elif mode == 'discover':
            self.rel_weights = nn.Parameter(torch.Tensor(self.nx, 1, self.nx))
        # init
        self._init_weights()
        
    def _init_weights(self):
        self.factors.data.uniform_(-0.1, 0.1)
        if self.mode == 'refine':
            self.rel_weights.data.fill_(0.5)
        elif self.mode == 'discover':
            self.rel_weights.data.fill_(1 / self.nx)
            
    def get_relations(self):
        if self.mode is None:
            return self.relations
        else:
            weights = F.hardtanh(self.rel_weights, 0, 1)
            if self.mode == 'refine':
                intra = self.rel_weights.new(self.nx, self.nx).copy_(self.relations[:, 0]).unsqueeze(1)
                inter = self.rel_weights.new_zeros(self.nx, self.nr - 1, self.nx)
                inter.masked_scatter_(self.relations[:, 1:], weights)
            if self.mode == 'discover':
                intra = self.relations[:, 0].unsqueeze(1)
                inter = weights
            return torch.cat((intra, inter), 1)
            
    def forward(self, batch):
        """
        调用模型计算这个batch输入对应的输出，nn.Module必须实现的接口
        :param batch: 输入数据，类字典，可以按字典的方法取数据
        :return: Zt+1
        a function get Zt and return Zt+1
        """
        # 1.取数据，假设字典中有4类数据，X,y,X_ext,y_ext
        # 当然一般只需要取输入数据，例如X,X_ext，因为这个函数是用来计算输出的
        # 模型输入的数据的特征维度应该等于self.feature_dim
        x =torch.Tensor(batch['X'])  # shape = (batch_size, input_length, ..., feature_dim)
        x_size = x.shape
        nowrel = self.get_relations()
        nowrel_size = nowrel.shape
        nowrel = nowrel.repeat(self.nt,1,1).expand(
        x_size[0],nowrel_size[0]*self.nt,nowrel_size[1],nowrel_size[2]) #64-12*41-2-41
        '''prrrr = open("oupppp.txt","w")
        print(nowrel.shape,file = prrrr)
        print((x_size[0]*self.nt*self.nx,nowrel_size[1],nowrel_size[2]),file = prrrr)
        prrrr.close()'''
        nowrel = nowrel.contiguous().view(
        x_size[0]*self.nt*self.nx,nowrel_size[1],nowrel_size[2]) #64*12*4-2-41
        z_inf = x.repeat(1,self.nx,1,1).view(
        x_size[0]*self.nt*self.nx,self.nx,self.nz) #64-12*41-41-1
        z_context =  nowrel.matmul(z_inf) #64*12*41-2-1
        z_gen = self.dynamic(z_context.view(-1,self.nr*self.nz))
        
        return self.activation(z_gen.view(x.shape))
        
        # 例如: y = batch['y'] / X_ext = batch['X_ext'] / y_ext = batch['y_ext']]
        # 2.根据输入数据计算模型的输出结果
        # 模型输出的结果的特征维度应该等于self.output_dim
        # 模型输出的结果的其他维度应该跟batch['y']一致，只有特征维度可能不同（因为batch['y']可能包含一些外部特征）
        # 如果模型的单步预测，batch['y']是多步的数据，则时间维度也有可能不同
        # 例如: outputs = self.model(x)
        # 3.返回输出结果
        # 例如: return outputs

    def calculate_loss(self, batch):
        """
        输入一个batch的数据，返回训练过程这个batch数据的loss，也就是需要定义一个loss函数。
        :param batch: 输入数据，类字典，可以按字典的方法取数据
        :return: training loss (tensor)
        """
        # 1.取出真值 ground_truth
        y_true = batch['y']
        # 2.取出预测值
        y_predicted = self.predict(batch)
        
        # 3.使用self._scaler将进行了归一化的真值和预测值进行反向归一化（必须）
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        # 4.调用loss函数计算真值和预测值的误差
        # trafficdl/model/loss.py中定义了常见的loss函数
        # 如果模型源码用到了其中的loss，则可以直接调用，以MSE为例:
        res = loss.masked_mse_torch(y_predicted, y_true)
        
        # 如果模型源码所用的loss函数在loss.py中没有，则需要自己实现loss函数
        # ...（自定义loss函数）
        # 5.返回loss的结果
        return res

    def predict(self, batch):
        """
        输入一个batch的数据，返回对应的预测值，一般应该是**多步预测**的结果
        一般会调用上边定义的forward()方法
        :param batch: 输入数据，类字典，可以按字典的方法取数据
        :return: predict result of this batch (tensor)
        """
        """
        包含三步：第一步，将真实数据Xt映射到状态Zt
                    第二步：利用Zt和forword向前计算12次，计算出Y所对应的状态集
                    第三步：对计算出的状态集进行解码，得到预测的Y
        """
        # 如果self.forward()的结果满足要求，可以直接返回
        # 如果不符合要求，例如self.forward()进行了单时间步的预测，但是模型训练时使用的是每个batch的数据进行的多步预测，
        # 则可以参考trafficdl/model/traffic_speed_prediction/STGCN.py中的predict()函数，进行多步预测
        # 多步预测的原则是: 先进行一步预测，用一步预测的结果进行二步预测，**而不是使用一步预测的真值进行二步预测!**
        # 以self.forward()的结果符合要求为例:
        x = torch.Tensor(batch['X'])  # shape = (batch_size, input_length, ..., feature_dim)
        x_size = x.shape
        #1 x_time_r = self.factf.matmul(x.view(self.nt,self.nx*x_size[0])).view(x_size)
        #1 x_loc_r = x_time_r.view(x_size[0]*self.nt,self.nx).matmul(self.facts).view(x_size)
        #1 z_inf = self.drop(x_loc_r)
        
        #step one:Xt to Zt
        x_stepone = self.ffunc(x.view(x_size[0],self.nt*self.nx*self.nz))
        x_steptwo = self.sigmo(x_stepone.view(x_size[0],self.nt,self.nx,self.nz*2))
        z_inf = self.drop(
        self.factors[(x_steptwo[:,:,:,0]*11).ceil().long(),(x_steptwo[:,:,:,1]*40).ceil().long()])
        
        batch['X'] = z_inf.view(x_size)
        #step two:Zt to Zt+12
        for i in range(self.input_window):
            z_next = self.forward(batch)
            batch['X'] = z_next
        z_inf = batch['X']
        #step three: Zt+12 to Y 
        x_rec = self.decoder(z_inf.view(-1,self.nz))
        return x_rec.view(x_size)