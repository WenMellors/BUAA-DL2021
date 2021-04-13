from logging import getLogger
import torch
from trafficdl.model import loss
from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel


class DGFN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        # 1.初始化父类
        super().__init__(config, data_feature)

    def forward(self, batch):

    def calculate_loss(self, batch):

    def predict(self, batch):
