from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from trafficdl.model import loss
import configargparse

class DotDict(dict):
    """
    Dot notation access to dictionary attributes
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class CRANN(AbstractTrafficStateModel):
    def __init__(self,config,data_feature):
        super().__init__(config,data_feature)
        ####

        p = configargparse.ArgParser()
        # -- data
        p.add('--datadir', type=str, help='path to dataset', default='data')
        p.add('--dataset', type=str, help='dataset name', default='dense_data.csv')
        # -- exp
        #p.add('--outputdir', type=str, help='path to save exp',
        #      default='output/' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        # -- model
        # ---- spatial module
        p.add('--dim_x', type=int, help='x dimension for image creation', default=30)
        p.add('--dim_y', type=int, help='x dimension for image creation', default=1)
        p.add('--n_inp_sp', type=int, help='number of input timesteps', default=12)
        p.add('--n_out_sp', type=int, help='number of output timesteps', default=12)
        p.add('--n_points', type=int, help='number of spatial points/sensors', default=30)
        # ---- temporal module
        p.add('--n_inp_tem', type=int, help='number of input timesteps', default=12 * 14)
        p.add('--n_out_tem', type=int, help='number of output timesteps', default=12)
        p.add('--in_dim_tem', type=int, help='number of input features', default=1)
        p.add('--out_dim_tem', type=int, help='number of output features', default=1)
        p.add('--n_hidden_tem', type=int, help='hidden dimension of enc-dec', default=100)
        p.add('--n_layers_tem', type=int, help='number of layers for enc-dec', default=1)
        # ---- dense module
        p.add('--n_exo', type=int, help='number of exogenous features', default=1)
        p.add('--n_hidden_dns', type=int, help='hidden dimension of dense', default=0)
        p.add('--n_layers_dns', type=int, help='number of layers for dense', default=1)
        p.add('--n_ar', type=int, help='number of autoregressive terms', default=4)
        # -- optim
        p.add('--lr', type=float, help='learning rate', default=1e-3)
        p.add('--beta1', type=float, default=.9, help='adam beta1')
        p.add('--beta2', type=float, default=.999, help='adam beta2')
        p.add('--eps', type=float, default=1e-8, help='adam eps')
        p.add('--wd', type=float, help='weight decay', default=5e-3)
        # -- learning
        p.add('--batch_size', type=int, default=64, help='batch size')
        p.add('--patience', type=int, default=10, help='number of epoch to wait before trigerring lr decay')
        p.add('--n_epochs', type=int, default=200, help='number of epochs to train for')
        # -- gpu
        p.add('--device', type=int, default=0, help='-1: cpu; > -1: cuda device id')

        # parse
        self.opt = DotDict(vars(p.parse_args()))

        ####

        # 2.从data_feature获取想要的信息，注意不同模型使用不同的Dataset类，其返回的data_feature内容不同（必须）
        # 以TrafficStateGridDataset为例演示取数据，可以取出如下的数据，用不到的可以不取
        # **这些参数的不能从config中取的**
        self._scaler = self.data_feature.get('scaler')  # 用于数据归一化
        self.adj_mx = self.data_feature.get('adj_mx', 1)  # 邻接矩阵
        self.num_nodes = self.data_feature.get('num_nodes', 1)  # 网格个数
        self.feature_dim = self.data_feature.get('feature_dim', 1)  # 输入维度
        self.output_dim = self.data_feature.get('output_dim', 1)  # 输出维度
        self.len_row = self.data_feature.get('len_row', 1)  # 网格行数
        self.len_column = self.data_feature.get('len_column', 1)  # 网格列数
        # 3.初始化log用于必要的输出（必须）
        self._logger = getLogger()
        # 4.初始化device（必须）
        self.device = config.get('device', torch.device('cpu'))
        # 5.初始化输入输出时间步的长度（非必须）
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)

        self.spatial_model = AttentionCNN(in_channels=self.opt.n_inp_sp, out_channels=self.opt.n_out_sp, dim_x=self.opt.dim_x, dim_y=self.opt.dim_y)
        self.temporal_encoder = EncoderLSTM(self.opt.in_dim_tem, self.opt.n_hidden_tem, device=self.device)
        self.temporal_decoder = BahdanauDecoder(self.opt.n_hidden_tem, self.opt.out_dim_tem)
        self.inputs = self.opt.n_out_sp * (self.opt.n_points + self.opt.n_exo + 1)
        self.outputs = self.opt.n_out_sp * self.opt.n_points
        self.model = MLP(n_inputs=self.inputs + self.opt.n_ar * self.opt.n_points, n_outputs=self.outputs, n_layers=self.opt.n_layers_dns,n_hidden=self.opt.n_hidden_dns)

    def evaluate_temp_att(self,encoder, decoder, batch, n_pred, device):
        """
        Inference of temporal attention mechanism
        """
        output = torch.Tensor().to(device)
        h = encoder.init_hidden(batch.size(0))
        encoder_output, h = encoder(batch, h)
        decoder_hidden = h
        decoder_input = torch.zeros(batch.size(0), 1, device=device)
        for k in range(n_pred):
            decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output)
            decoder_input = decoder_output
            output = torch.cat((output, decoder_output), 1)
        return output

    def forward(self,batch):
        x0 = batch.data['X']
        x_time = x0.reshape(x0.size()[0],x0.size()[1]*x0.size()[2]*x0.size()[3])
        x_space = x0
        x_exo = x0[:,:,:1,:].squeeze(2)
        y = batch.data['y']
        if torch.cuda.is_available():
            x_time = x_time.view(64, -1, 1).to(self.device)
            x_space = x_space.to(self.device)
            x_exo = x_exo.to(self.device)
            y = y.to(self.device)
        else:
            x_time = x_time
            x_space = x_space
            x_exo = x_exo
            y = y
        y_time = self.evaluate_temp_att(self.temporal_encoder, self.temporal_decoder, x_time, self.opt.n_out_sp, self.device)
        y_space = self.spatial_model(x_space)[0]
        x = torch.cat((y_time.unsqueeze(2), y_space.squeeze().view(-1,self.opt.n_out_sp,self.opt.n_points), x_exo), dim = 2).view(-1, self.inputs)
        x = torch.cat((x,x_space[:,-self.opt.n_ar:].view(-1,self.opt.n_ar*self.opt.n_points)), dim = 1)
        y_pred = self.model(x).view(-1,self.opt.n_out_sp,self.opt.dim_x,self.opt.dim_y)
        return y_pred





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
        return self.forward(batch)

#S model
class AttentionCNN(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Spatial module with spatio-temporal attention

    --------------
    | Attributes |
    --------------
    in_channels : int
        Number of input timesteps
    out_channels : int
        Number of output timesteps
    dim_x : int
        Dimension of x-axis for input images
    dim_y : int
        Dimension of y-axis for input images

    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """

    def __init__(self, in_channels, out_channels, dim_x, dim_y):
        super(AttentionCNN, self).__init__()
        # Variables
        self.out_channels = out_channels
        self.dim_x = dim_x
        self.dim_y = dim_y

        # Conv blocks
        self.conv_block1 = ConvBlock(in_channels, 64, 5)

        # Attention
        self.att1 = AttentionBlock(dim_x, dim_y, 12, method='hadamard')

        # Output
        self.regressor = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.regressor(out)
        out, att = self.att1(out)
        return out, att


class ConvBlock(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Convolutional blocks of num_conv convolutions with out_features channels

    --------------
    | Attributes |
    --------------
    in_features : int
        Number of input channels
    out_features : int
        Number of middle and output channels
    num_conv : int
        Number of convolutions

    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """

    def __init__(self, in_features, out_features, num_conv):
        super(ConvBlock, self).__init__()
        features = [in_features] + [out_features for i in range(num_conv)]
        layers = []
        for i in range(len(features) - 1):
            layers.append(
                nn.Conv2d(in_channels=features[i], out_channels=features[i + 1], kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i + 1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)


class AttentionBlock(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Attentional block for spatio-temporal attention mechanism

    --------------
    | Attributes |
    --------------
    dim_x : int
        Dimension of x-axis for input images
    dim_y : int
        Dimension of y-axis for input images
    timesteps : int
        Number of input timesteps
    method : str
        Attentional function to calculate attention weights

    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """

    def __init__(self, dim_x, dim_y, timesteps, method='hadamard'):
        super(AttentionBlock, self).__init__()
        # Variables
        self.method = method
        self.weight = nn.Parameter(torch.FloatTensor(timesteps, dim_x * dim_y, dim_x * dim_y))
        torch.nn.init.xavier_uniform_(self.weight)
        if method == 'general':
            self.fc = nn.Linear(timesteps * (dim_x * dim_y) ** 2, timesteps * (dim_x * dim_y) ** 2, bias=False)
        elif method == 'concat':
            self.fc = nn.Linear(timesteps * (dim_x * dim_y) ** 2, timesteps * (dim_x * dim_y) ** 2, bias=False)

    def forward(self, x, y=0):
        N, T, W, H = x.size()
        if self.method == 'hadamard':
            xp = x.view(N, T, -1).repeat(1, 1, W * H).view(N, T, W * H, W * H)
            wp = self.weight.expand_as(xp)
            alig_scores = wp.mul(xp)
        elif self.method == 'general':
            xp = x.view(N, T, -1).repeat(1, 1, W * H).view(N, T, W * H, W * H)
            wp = self.weight.expand_as(xp)
            alig_scores = self.fc((wp.mul(xp)).view(N, -1))
        elif self.method == 'concat':
            xp = x.view(N, T, -1).repeat(1, 1, W * H).view(N, T, W * H, W * H)
            wp = self.weight.expand_as(xp)
            alig_scores = torch.tanh(self.fc((wp + xp).view(N, -1)))
        elif self.method == 'dot':
            xp = x.view(N, T, -1).repeat(1, 1, W * H).view(N, T, W * H, W * H)
            alig_scores = self.weight.matmul(xp)

        att_weights = F.softmax(alig_scores.view(N, T, W * H, W * H), dim=3)
        out = att_weights.matmul(x.view(N, T, -1).unsqueeze(3))
        return out.view(N, T, W, H), att_weights

#T model
class EncoderLSTM(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Encoder for temporal module

    --------------
    | Attributes |
    --------------
    input_size : int
        Number of input features
    hidden_size : int
        Dimension of hidden space
    n_layers : int
        Number of layers for the encoder
    drop_prob : float
        Dropout for the encoder
    device : int/str
        Device in which hiddens are stored

    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """

    def __init__(self, input_size, hidden_size, n_layers=1, drop_prob=0, device='cuda'):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=drop_prob, batch_first=True)

    def forward(self, inputs, hidden):
        output, hidden = self.lstm(inputs, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device))


class BahdanauDecoder(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Decoder an attention mechanism for temporal module

    --------------
    | Attributes |
    --------------
    hidden_size : int
        Dimension of hidden space
    output_size : int
        Number of output features
    n_layers : int
        Number of layers for the encoder
    drop_prob : float
        Dropout for the encoder

    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """

    def __init__(self, hidden_size, output_size, n_layers=1, drop_prob=0.1):
        super(BahdanauDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
        torch.nn.init.xavier_uniform_(self.weight)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size + self.output_size, self.hidden_size, batch_first=True)
        self.fc_prediction = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.squeeze()

        # Calculating Alignment Scores
        x = torch.tanh(self.fc_hidden(hidden[0].view(-1, 1, self.hidden_size)) +
                       self.fc_encoder(encoder_outputs))

        alignment_scores = x.matmul(self.weight.unsqueeze(2))

        # Softmaxing alignment scores to get Attention weights
        attn_weights = F.softmax(alignment_scores.view(inputs.size(0), -1), dim=1)

        # Multiplying the Attention weights with encoder outputs to get the context vector
        self.context_vector = torch.matmul(attn_weights.unsqueeze(1), encoder_outputs)

        # Concatenating context vector with embedded input word
        output = torch.cat((inputs, self.context_vector.squeeze(1)), 1).unsqueeze(1)
        # Passing the concatenated vector as input to the LSTM cell
        output, hidden = self.lstm(output, hidden)

        output = self.fc_prediction(output).squeeze(2)

        return output, hidden, attn_weights

#D model
class MLP(nn.Module):
    """
    ---------------
    | Description |
    ---------------
    Dense module

    --------------
    | Attributes |
    --------------
    n_inputs : int
        Number of input features
    n_outputs : int
        Number of output features
    n_layers : int
        Number of layers
    n_hidden : int
        Dimension of hidden layers

    -----------
    | Methods |
    -----------
    forward(x)
        Forward pass of the network
    """

    def __init__(self, n_inputs, n_outputs, n_layers=1, n_hidden=0, dropout=0):
        super(MLP, self).__init__()
        if n_layers < 1:
            raise ValueError('Number of layers needs to be at least 1.')
        elif n_layers == 1:
            self.module = nn.Linear(n_inputs, n_outputs)
        else:
            modules = [nn.Linear(n_inputs, n_hidden), nn.ReLU(), nn.Dropout(dropout)]
            n_layers -= 1
            while n_layers > 1:
                modules += [nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Dropout(dropout)]
                n_layers -= 1
            modules.append(nn.Linear(n_hidden, n_outputs))
            self.module = nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)