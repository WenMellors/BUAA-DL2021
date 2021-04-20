import torch
import torch.nn as nn
from logging import getLogger
from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel
from trafficdl.model import loss


class unit(nn.Module):
    def __init__(self, in_c, out_c, pool=False):
        super(unit, self).__init__()
        self.pool1 = nn.MaxPool2d((2, 2), padding=(0, 1))
        self.conv1 = nn.Conv2d(in_c, out_c, 1, 2, 0)

        self.bn1 = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_c, out_c, 3, 1, 1)

        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_c, out_c, 3, 1, 1)

        self.pool = pool

    def forward(self, x):
        res = x
        if self.pool:
            x = self.pool1(x)
            res = self.conv1(res)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = res+out
        return out


class attention_3d_block(nn.Module):
    def __init__(self):
        super(attention_3d_block, self).__init__()
        self.fc = nn.Linear(1, 276)

    def forward(self, x_):
        x = x_.permute(0, 2, 1)
        x = self.fc(x)
        x_probs = x.permute(0, 2, 1)
        # x = x.unsqueeze(1)
        # x_probs = x_probs.unsqueeze(1)
        xx = torch.matmul(x, x_probs)

        return xx


class ResLSTM(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._scaler = self.data_feature.get('scaler')
        self.adj_mx = self.data_feature.get('adj_mx')
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv1_1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.unit1 = unit(32, 32)
        self.unit2 = unit(32, 64, True)
        self.fc1 = nn.Linear(26496, 276)

        self.fc2 = nn.Linear(55, 276)
        self.lstm1 = nn.LSTM(input_size=276, hidden_size=128, num_layers=2)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=276, num_layers=2)
        self.fc3 = nn.Linear(276, 276)

        self.lstm3 = nn.LSTM(input_size=276, hidden_size=128, num_layers=2)
        self.att = attention_3d_block()
        self.fc_last = nn.Linear(128*128, 276)

    def conv_block1(self, x):
        x = self.conv1(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

    def conv_block2(self, x):
        x = self.conv1_1(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

    def fourth_pro(self, x):
        x = x.contiguous().view(x.size()[0], -1)
        x = self.fc2(x)
        x = x.view(x.size()[0], 1, 276)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc3(x)
        return x

    def forward(self, batch):
        x1 = batch.data["X"][:, :, :, :3]
        x1 = x1.permute(0, 3, 2, 1)
        x2 = batch.data["X"][:, :, :, :3]
        x2 = x2.permute(0, 3, 2, 1)
        x3 = batch.data["X"][:, :, :, :1].permute(0, 3, 2, 1)
        x4 = batch.data["X"][:, :, :1, 3:].permute(0, 2, 3, 1)

        p1 = self.conv_block1(x1)
        p2 = self.conv_block1(x2)
        p3 = self.conv_block2(x3)
        p4 = self.fourth_pro(x4)

        out = p1+p2+p3+p4
        out = out.view(out.size()[0], 1, 276)
        out, _ = self.lstm3(out)
        out = self.att(out)
        out = out.view(out.size()[0], -1)
        out = self.fc_last(out)
        return out.permute(1, 0)

    def calculate_loss(self, batch, batches_seen=None):
        y_true = batch['y']
        y_predicted = self.predict(batch)  # prediction results
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)
