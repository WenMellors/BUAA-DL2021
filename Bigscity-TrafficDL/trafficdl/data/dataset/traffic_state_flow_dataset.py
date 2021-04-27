import os
from trafficdl.data.dataset.master.utils.tools import DatasetGenerator
from trafficdl.data.my_dataset import MyDataset
from trafficdl.data.utils import generate_dataloader
from trafficdl.model.traffic_flow_prediction.data_parameters import data_parameters
from trafficdl.utils import ensure_dir


class TrafficStateFlowDataset():

    def __init__(self, config):     # 调用父类的初始化参数， 里面有好多config中的参数
        # 模型参数
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.batch_size = self.config.get('batch_size', 64)
        self.cache_dataset = self.config.get('cache_dataset', True)
        self.num_workers = self.config.get('num_workers', 0)
        self.pad_with_last_sample = self.config.get(
            'pad_with_last_sample', True)
        self.train_rate = self.config.get('train_rate', 0.7)
        self.eval_rate = self.config.get('eval_rate', 0.1)
        self.scaler_type = self.config.get('scaler', 'none')
        self.load_external = self.config.get('load_external', False)
        self.normal_external = self.config.get('normal_external', False)
        self.input_window = self.config.get('input_window', 12)
        self.output_window = self.config.get('output_window', 12)
        self.output_dim = self.config.get('output_dim', 0)
        self.add_time_in_day = self.config.get('add_time_in_day', False)
        self.add_day_in_week = self.config.get('add_day_in_week', False)
        self.calculate_weight = self.config.get('calculate_weight', False)
        self.adj_epsilon = self.config.get('adj_epsilon', 0.1)
        self.parameters_str = \
            str(self.dataset) + '_' + str(self.input_window) + '_' + str(self.output_window) + '_' \
            + str(self.train_rate) + '_' + str(self.eval_rate) + '_' + str(self.scaler_type) + '_' \
            + str(self.batch_size) + '_' + str(self.add_time_in_day) + '_' \
            + str(self.add_day_in_week) + '_' + str(self.pad_with_last_sample)
        self.cache_file_name = os.path.join(
            './trafficdl/cache/dataset_cache/',
            'traffic_state_{}.npz'.format(
                self.parameters_str))
        self.cache_file_folder = './trafficdl/cache/dataset_cache/'
        ensure_dir(self.cache_file_folder)

        # 加载这个模型所特有的参数
        self.d_model = config.get("d_model", 64)
        self.n_w = config.get("n_w", 1)
        self.n_d = config.get("n_d", 3)
        self.n_wd_times = config.get("w_wd_times", 1)
        self.n_p = config.get("n_p", 1)
        self.n_before = config.get("n_before", 0)
        self.n_pred = config.get("n_pred", 12)
        self.l_half = config.get("l_half", 3)
        self.l_half_g = config.get("l_half_g", 5)
        self.pre_shuffle = config.get("pre_shuffle", True)
        self.same_padding = config.get("same_padding", False)
        self.test_model = config.get("test_model", None)
        self.no_save = config.get("no_save", False)
        self.load_saved_data = config.get("load_saved_data", True)
        self.st_revert = config.get("st_revert", False)
        # 设置data_parameter
        param = data_parameters[self.dataset]
        self.param = param

        # self.strategy = tf.distribute.MirroredStrategy()
        # 初始化
        self.data = None
        # 这个数据的类型是否有问题
        self.feature_name = {"dae_inp_g": "float",
                             "dae_inp": "float",
                             "dae_inp_ex": "float",
                             "sad_inp": "float",
                             "sad_inp_ex": "float",
                             "cors": "float",
                             "cors_g": "float",
                             "y": "float"
                             }  # 这个需要设置所有的标签
        # "threshold_mask_g": "float",
        # "threshold_mask": "float",
        # "combined_mask": "float",
        self.adj_mx = None
        self.scaler = None
        self.feature_dim = 0
        self.num_nodes = 0

    # 获取数据
    def get_data(self):
        """
                return:
                    train_dataloader (pytorch.DataLoader)
                    eval_dataloader (pytorch.DataLoader)
                    test_dataloader (pytorch.DataLoader)
                    all the dataloaders are composed of Batch (class)
        """
        # strategy = self.strategy
        self.GLOBAL_BATCH_SIZE = self.batch_size
        # * strategy.num_replicas_in_sync
        self.dataset_generator = DatasetGenerator(self.d_model,
                                                  self.dataset,
                                                  self.GLOBAL_BATCH_SIZE,
                                                  self.n_w,
                                                  self.n_d,
                                                  self.n_wd_times,
                                                  self.n_p,
                                                  self.n_before,
                                                  self.n_pred,
                                                  self.l_half,
                                                  self.l_half_g,
                                                  self.pre_shuffle,
                                                  self.same_padding,
                                                  self.test_model
                                                  )

        dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, y = \
            self.dataset_generator.build_dataset('train',
                                                 self.load_saved_data, None, self.st_revert, self.no_save)

        # 这里只需要load一下， 只load对应的train文件夹，然后会被切分成三个部分

        train_data = MyDataset(
            dae_inp_g,
            dae_inp,
            dae_inp_ex,
            sad_inp,
            sad_inp_ex,
            cors,
            cors_g,
            y,
            self.param['pred_type'])

        dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, y = \
            self.dataset_generator.build_dataset('val', self.load_saved_data, None, self.st_revert, self.no_save)

        eval_data = MyDataset(
            dae_inp_g,
            dae_inp,
            dae_inp_ex,
            sad_inp,
            sad_inp_ex,
            cors,
            cors_g,
            y,
            self.param['pred_type'])

        dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, y = \
            self.dataset_generator.build_dataset('test',
                                                 self.load_saved_data, None, self.st_revert, self.no_save)

        test_data = MyDataset(
            dae_inp_g,
            dae_inp,
            dae_inp_ex,
            sad_inp,
            sad_inp_ex,
            cors,
            cors_g,
            y,
            self.param['pred_type'])

        # train_data = MyDataset(self.dataset_generator.dae_inp_g[0], self.dataset_generator.dae_inp[0],
        #                        self.dataset_generator.dae_inp_ex[0], self.dataset_generator.sad_inp[0],
        #                        self.dataset_generator.sad_inp_ex[0], self.dataset_generator.cors[0],
        #                        self.dataset_generator.cors_g[0],
        #                        self.dataset_generator.y[0], self.param['pred_type'])
        # eval_data = MyDataset(self.dataset_generator.dae_inp_g[1], self.dataset_generator.dae_inp[1],
        #                        self.dataset_generator.dae_inp_ex[1], self.dataset_generator.sad_inp[1],
        #                        self.dataset_generator.sad_inp_ex[1], self.dataset_generator.cors[1],
        #                        self.dataset_generator.cors_g[1],
        #                        self.dataset_generator.y[1], self.param['pred_type'])
        # test_data = MyDataset(self.dataset_generator.dae_inp_g[2],
        # self.dataset_generator.dae_inp[2],
        #                        self.dataset_generator.dae_inp_ex[2], self.dataset_generator.sad_inp[2],
        #                        self.dataset_generator.sad_inp_ex[2], self.dataset_generator.cors[2],
        # self.dataset_generator.cors_g[2], self.dataset_generator.y[2],
        # self.param['pred_type'])

        print("train_data")
        print(len(train_data))
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers,
                                pad_with_last_sample=self.pad_with_last_sample)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """
                返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
                             feature_dim是输入数据的维度，output_dim是模型输出的维度
                :return: data_feature (dict)
        """
        return {"scaler": self.scaler, "adj_mx": self.adj_mx,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim}
