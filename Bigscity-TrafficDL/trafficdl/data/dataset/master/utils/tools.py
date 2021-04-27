from __future__ import absolute_import, division, print_function, unicode_literals
from trafficdl.data.dataset.master.utils.DataLoader import DataLoader as Dl
# from utils.DataLoader import DataLoader as dl


class DatasetGenerator:
    def __init__(
            self,
            d_model=64,
            dataset='taxi',
            batch_size=64,
            n_w=1,
            n_d=3,
            n_wd_times=1,
            n_p=1,
            n_before=0,
            n_pred=12,
            l_half=3,
            l_half_g=None,
            pre_shuffle=True,
            same_padding=False,
            test_model=False):
        self.d_model = d_model
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_w = n_w
        self.n_d = n_d
        self.n_wd_times = n_wd_times
        self.n_p = n_p
        self.n_before = n_before
        self.n_pred = n_pred
        self.l_half = l_half
        self.l_half_g = l_half_g
        self.pre_shuffle = pre_shuffle
        self.same_padding = same_padding
        self.test_model = test_model

        self.val_set = None
        self.test_set = None

        self.dae_inp_g = []
        self.dae_inp = []
        self.dae_inp_ex = []
        self.sad_inp = []
        self.sad_inp_ex = []
        self.cors = []
        self.cors_g = []
        self.y = []

    # load数据
    # load_saved_data 决定是否处理已加载的数据，还是重新进行load
    def load_data(
            self,
            datatype,
            load_saved_data=True,
            st_revert=False,
            no_save=False):
        data_loader = Dl(
            self.d_model,
            self.dataset,
            self.l_half,
            self.l_half_g,
            self.pre_shuffle,
            self.same_padding,
            self.test_model)
        self.dae_inp_g, self.dae_inp, self.dae_inp_ex, self.sad_inp, \
            self.sad_inp_ex, self.cors, self.cors_g, self.y = data_loader.generate_data(
                datatype, self.n_w, self.n_d, self.n_wd_times,
                self.n_p, self.n_before, self.n_pred, load_saved_data,
                st_revert, no_save)

    # load_saved_data 是否load已训练好的数据,这里的数据已经进行了划分，划分成为了三个部分，train，val，test

    def build_dataset(
            self,
            datatype='train',
            load_saved_data=True,
            strategy=None,
            st_revert=False,
            no_save=None):
        assert datatype in ['train', 'val', 'test']
        self.load_data(datatype, load_saved_data, st_revert, no_save)
        return self.dae_inp_g, self.dae_inp, self.dae_inp_ex, self.sad_inp, \
            self.sad_inp_ex, self.cors, self.cors_g, self.y
