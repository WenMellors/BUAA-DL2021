import numpy as np
# from utils.CordinateGenerator import CordinateGenerator

# from data_parameters import data_parameters
from trafficdl.data.dataset.master.utils.data_parameters import data_parameters


class DataLoader:
    def __init__(
            self,
            d_model,
            dataset='taxi',
            l_half=3,
            l_half_g=None,
            pre_shuffle=True,
            same_padding=False,
            test_model=None):
        assert dataset in ['taxi', 'bike', 'ctm']
        self.dataset = dataset
        self.pmt = data_parameters[dataset]  # data_parameters里面存储着所有的文件
        self.l_half = l_half
        self.l_half_g = l_half_g
        # self.cor_gen = CordinateGenerator(self.pmt['len_r'], self.pmt['len_c'], d_model, l_half=l_half)
        # self.cor_gen_g = CordinateGenerator(self.pmt['len_r'], self.pmt['len_c'], d_model, l_half=l_half_g)
        self.pre_shuffle = pre_shuffle
        self.same_padding = same_padding
        self.test_model = test_model

    # 如果是第一次loaddata需要调用这个数据
    def load_data(self, datatype='train'):
        pred_type = self.pmt['pred_type']
        data_max = self.pmt['data_max']
        if datatype == 'train':
            data = np.load(self.pmt['data_train'])
        elif datatype == 'val':
            data = np.load(self.pmt['data_val'])
        else:
            data = np.load(self.pmt['data_test'])

        if self.dataset in ['taxi', 'bike']:
            self.data_mtx = np.array(
                data['flow'], dtype=np.float32) / np.array(data_max[:pred_type], dtype=np.float32)
            self.t_mtx = np.array(
                data['trans'], dtype=np.float32) / np.array(data_max[pred_type:], dtype=np.float32)
        else:
            self.data_mtx = np.array(data['data'], dtype=np.float32)
            self.data_mtx = self.data_mtx / \
                np.array(data_max, dtype=np.float32)

        self.ex_mtx = data['ex_knlg']

    def generate_data(
            self,
            datatype='train',
            n_w=1,
            n_d=3,
            n_wd_times=1,
            n_p=1,
            n_before=0,
            n_pred=12,
            load_saved_data=False,
            st_revert=False,
            no_save=False):

        assert datatype in ['train', 'val', 'test']

        dae_inp_g = []
        dae_inp = []
        dae_inp_ex = []

        sad_inp = []
        sad_inp_ex = []

        cors_g = []
        cors = []

        y = []

        """ loading saved data """
        # trafficdl/data/dataset/master/rawData, 数据所在的相对路径
        if load_saved_data and not self.test_model:
            print('Loading {} data from .npzs...'.format(datatype))
            dae_inp_g = np.load(
                "trafficdl/cache/dataset_cache/dae_inp_g_{}_{}.npz".format(self.dataset, datatype))['data']
            dae_inp = np.load(
                "trafficdl/cache/dataset_cache/dae_inp_{}_{}.npz".format(self.dataset, datatype))['data']
            dae_inp_ex = np.load(
                "trafficdl/cache/dataset_cache/dae_inp_ex_{}_{}.npz".format(self.dataset, datatype))['data']
            sad_inp = np.load(
                "trafficdl/cache/dataset_cache/sad_inp_{}_{}.npz".format(self.dataset, datatype))['data']
            sad_inp_ex = np.load(
                "trafficdl/cache/dataset_cache/sad_inp_ex_{}_{}.npz".format(self.dataset, datatype))['data']
            cors = np.load(
                "trafficdl/cache/dataset_cache/cors_{}_{}.npz".format(self.dataset, datatype))['data']
            cors_g = np.load(
                "trafficdl/cache/dataset_cache/cors_g_{}_{}.npz".format(self.dataset, datatype))['data']
            y = np.load(
                "trafficdl/cache/dataset_cache/y_{}_{}.npz".format(self.dataset, datatype))['data']
        # else:
        #     l_half = self.l_half
        #     l_half_g = self.l_half_g
        #
        #     print("Loading {} data...".format(datatype))
        #     """ loading data """
        #     self.load_data(datatype)
        #
        #     data_mtx = self.data_mtx
        #     ex_mtx = self.ex_mtx
        #     data_shape = data_mtx.shape
        #     crowd_flow = self.dataset in ['taxi', 'bike']
        #     if crowd_flow:
        #         t_mtx = self.t_mtx
        #
        #     if l_half:
        #         l_full = 2 * l_half + 1
        #
        #     if l_half_g:
        #         l_full_g = 2 * l_half_g + 1
        #
        #     """ initialize the array to hold the final inputs """
        #
        #     dae_inp_g = []
        #     dae_inp = []
        #     dae_inp_ex = []
        #
        #     sad_inp = []
        #     sad_inp_ex = []
        #
        #     cors_g = []
        #     cors = []
        #
        #     y = []
        #
        #     assert n_w >= 0 and n_d >= 0 and n_d < 7
        #     """ set the start time interval to sample the data"""
        #     s1 = n_d * self.pmt['n_int'] + n_before
        #     s2 = n_w * 7 * self.pmt['n_int'] + n_before
        #     time_start = max(s1, s2)
        #     time_end = data_shape[0] - n_pred
        #
        #     for t in range(time_start, time_end):
        #         if (t - time_start) % 100 == 0:
        #             print("Loading {}/{}".format(t - time_start, time_end - time_start))
        #
        #         for r in range(data_shape[1]):
        #             for c in range(data_shape[2]):
        #
        #                 """ initialize the array to hold the samples of each node at each time interval """
        #
        #                 dae_inp_g_sample = []
        #                 dae_inp_sample = []
        #                 dae_inp_ex_sample = []
        #
        #                 if l_half:
        #                     """ initialize the boundaries of the area of interest """
        #                     r_start = r - l_half  # the start location of each AoI
        #                     c_start = c - l_half
        #
        #                     """ adjust the start location if it is on the boundaries of the grid map """
        #                     if r_start < 0:
        #                         r_start_l = 0 - r_start
        #                         r_start = 0
        #                     else:
        #                         r_start_l = 0
        #                     if c_start < 0:
        #                         c_start_l = 0 - c_start
        #                         c_start = 0
        #                     else:
        #                         c_start_l = 0
        #
        #                     r_end = r + l_half + 1  # the end location of each AoI
        #                     c_end = c + l_half + 1
        #                     if r_end >= data_shape[1]:
        #                         r_end_l = l_full - (r_end - data_shape[1])
        #                         r_end = data_shape[1]
        #                     else:
        #                         r_end_l = l_full
        #                     if c_end >= data_shape[2]:
        #                         c_end_l = l_full - (c_end - data_shape[2])
        #                         c_end = data_shape[2]
        #                     else:
        #                         c_end_l = l_full
        #
        #                 if l_half_g:
        #                     """ initialize the boundaries of the area of interest """
        #                     r_start_g = r - l_half_g  # the start location of each AoI
        #                     c_start_g = c - l_half_g
        #
        #                     """ adjust the start location if it is on the boundaries of the grid map """
        #                     if r_start_g < 0:
        #                         r_start_g_l = 0 - r_start_g
        #                         r_start_g = 0
        #                     else:
        #                         r_start_g_l = 0
        #                     if c_start_g < 0:
        #                         c_start_g_l = 0 - c_start_g
        #                         c_start_g = 0
        #                     else:
        #                         c_start_g_l = 0
        #
        #                     r_end_g = r + l_half_g + 1  # the end location of each AoI
        #                     c_end_g = c + l_half_g + 1
        #                     if r_end_g >= data_shape[1]:
        #                         r_end_g_l = l_full_g - (r_end_g - data_shape[1])
        #                         r_end_g = data_shape[1]
        #                     else:
        #                         r_end_g_l = l_full_g
        #                     if c_end_g >= data_shape[2]:
        #                         c_end_g_l = l_full_g - (c_end_g - data_shape[2])
        #                         c_end_g = data_shape[2]
        #                     else:
        #                         c_end_g_l = l_full_g
        #
        #                 """ start the samplings of previous weeks """
        #                 t_hist = []
        #
        #                 for week_cnt in range(n_w):
        #                     s_time_w = int(t - (n_w - week_cnt) * 7 * self.pmt['n_int'] - n_before)
        #
        #                     for int_cnt in range(n_wd_times):
        #                         t_hist.append(s_time_w + int_cnt)
        #
        #                 """ start the samplings of previous days"""
        #                 for hist_day_cnt in range(n_d):
        #                     """ define the start time in previous days """
        #                     s_time_d = int(t - (n_d - hist_day_cnt)
        #                     * self.pmt['n_int'] - n_before)
        #
        #                     """ generate samples from the previous days """
        #                     for int_cnt in range(n_wd_times):
        #                         t_hist.append(s_time_d + int_cnt)
        #
        #                 """ sampling of inputs of current day,
        #                 the details are similar to those mentioned above """
        #                 for int_cnt in range(n_p):
        #                     t_hist.append(t - n_p + int_cnt)
        #
        #                 for t_now in t_hist:
        #                     if not l_half_g:
        #                         one_inp_g = data_mtx[t_now, ...]
        #
        #                         if crowd_flow:
        #                             one_inp_g_t = np.zeros((data_shape[1], data_shape[2], 2), dtype=np.float32)
        #
        #                             one_inp_g_t[..., 0] += t_mtx[t_now, ..., r, c, 0]
        #                             one_inp_g_t[..., 0] += t_mtx[t_now, ..., r, c, 1]
        #                             one_inp_g_t[..., 1] += t_mtx[t_now, r, c, ..., 0]
        #                             one_inp_g_t[..., 1] += t_mtx[t_now, r, c, ..., 1]
        #
        #                     else:
        #                         one_inp_g = np.zeros((l_full_g, l_full_g, 2), dtype=np.float32)
        #                         one_inp_g[r_start_g_l:r_end_g_l, c_start_g_l:c_end_g_l, :] = \
        #                             data_mtx[t_now, r_start_g:r_end_g, c_start_g:c_end_g, :]
        #
        #                         if crowd_flow:
        #                             one_inp_g_t = np.zeros((l_full_g, l_full_g, 2), dtype=np.float32)
        #                             one_inp_g_t[r_start_g_l:r_end_g_l, c_start_g_l:c_end_g_l, 0] += \
        #                                 t_mtx[t_now, r_start_g:r_end_g, c_start_g:c_end_g, r, c, 0]
        #                             one_inp_g_t[r_start_g_l:r_end_g_l, c_start_g_l:c_end_g_l, 0] += \
        #                                 t_mtx[t_now, r_start_g:r_end_g, c_start_g:c_end_g, r, c, 1]
        #                             one_inp_g_t[r_start_g_l:r_end_g_l, c_start_g_l:c_end_g_l, 1] += \
        #                                 t_mtx[t_now, r, c, r_start_g:r_end_g, c_start_g:c_end_g, 0]
        #                             one_inp_g_t[r_start_g_l:r_end_g_l, c_start_g_l:c_end_g_l, 1] += \
        #                                 t_mtx[t_now, r, c, r_start_g:r_end_g, c_start_g:c_end_g, 1]
        #
        #                     dae_inp_g_sample.append(
        #                         np.concatenate([one_inp_g, one_inp_g_t], axis=-1) if crowd_flow else one_inp_g)
        #
        #                     if not l_half:
        #                         one_inp = data_mtx[t_now, ...]
        #
        #                         if crowd_flow:
        #                             one_inp_t = np.zeros((data_shape[1], data_shape[2], 2), dtype=np.float32)
        #
        #                             one_inp_t[..., 0] += t_mtx[t_now, ..., r, c, 0]
        #                             one_inp_t[..., 0] += t_mtx[t_now, ..., r, c, 1]
        #                             one_inp_t[..., 1] += t_mtx[t_now, r, c, ..., 0]
        #                             one_inp_t[..., 1] += t_mtx[t_now, r, c, ..., 1]
        #
        #                     else:
        #                         one_inp = np.zeros((l_full, l_full, 2), dtype=np.float32)
        #                         if self.same_padding:
        #                             one_inp[...] = data_mtx[t_now, r, c, :]
        #                         one_inp[r_start_l:r_end_l, c_start_l:c_end_l, :] = \
        #                             data_mtx[t_now, r_start:r_end, c_start:c_end, :]
        #
        #                         if crowd_flow:
        #                             one_inp_t = np.zeros((l_full, l_full, 2), dtype=np.float32)
        #                             one_inp_t[r_start_l:r_end_l, c_start_l:c_end_l, 0] += \
        #                                 t_mtx[t_now, r_start:r_end, c_start:c_end, r, c, 0]
        #                             one_inp_t[r_start_l:r_end_l, c_start_l:c_end_l, 0] += \
        #                                 t_mtx[t_now, r_start:r_end, c_start:c_end, r, c, 1]
        #                             one_inp_t[r_start_l:r_end_l, c_start_l:c_end_l, 1] += \
        #                                 t_mtx[t_now, r, c, r_start:r_end, c_start:c_end, 0]
        #                             one_inp_t[r_start_l:r_end_l, c_start_l:c_end_l, 1] += \
        #                                 t_mtx[t_now, r, c, r_start:r_end, c_start:c_end, 1]
        #
        #                     dae_inp_sample.append(
        #                         np.concatenate([one_inp, one_inp_t], axis=-1) if crowd_flow else one_inp)
        #                     dae_inp_ex_sample.append(ex_mtx[t_now, :])
        #
        #                 """ append the samples of each node to the overall inputs arrays """
        #                 dae_inp_g.append(dae_inp_g_sample)
        #                 dae_inp.append(dae_inp_sample)
        #                 dae_inp_ex.append(dae_inp_ex_sample)
        #
        #                 sad_inp.append(data_mtx[t - 1: t + n_pred - 1, r, c, :])
        #                 sad_inp_ex.append(ex_mtx[t - 1: t + n_pred - 1, :])
        #
        #                 cors.append(self.cor_gen.get(r, c))
        #                 cors_g.append(self.cor_gen_g.get(r, c))
        #
        #                 """ generating the ground truth for each sample """
        #                 y.append(data_mtx[t: t + n_pred, r, c, :])
        #
        #         if self.test_model and t + 1 - time_start >= self.test_model:
        #             break
        #
        #     """ convert the inputs arrays to matrices """
        #     dae_inp_g = np.array(dae_inp_g, dtype=np.float32)
        #     dae_inp = np.array(dae_inp, dtype=np.float32)
        #     dae_inp_ex = np.array(dae_inp_ex, dtype=np.float32)
        #
        #     sad_inp = np.array(sad_inp, dtype=np.float32)
        #     sad_inp_ex = np.array(sad_inp_ex, dtype=np.float32)
        #
        #     cors = np.array(cors, dtype=np.float32)
        #     cors_g = np.array(cors_g, dtype=np.float32)
        #
        #     y = np.array(y, dtype=np.float32)
        #
        #     if st_revert:
        #         dae_inp_g = dae_inp_g.transpose((0, 2, 3, 1, 4))
        #         dae_inp = dae_inp.transpose((0, 2, 3, 1, 4))
        #
        #     """ save the matrices """
        #     if not (self.test_model or no_save):
        #         print('Saving .npz files...')
        #         np.savez_compressed("data/dae_inp_g_{}_{}.npz".format(self.dataset, datatype), data=dae_inp_g)
        #         np.savez_compressed("data/dae_inp_{}_{}.npz".format(self.dataset, datatype), data=dae_inp)
        #         np.savez_compressed("data/dae_inp_ex_{}_{}.npz".format(self.dataset, datatype), data=dae_inp_ex)
        #         np.savez_compressed("data/sad_inp_{}_{}.npz".format(self.dataset, datatype), data=sad_inp)
        #         np.savez_compressed("data/sad_inp_ex_{}_{}.npz".format(self.dataset, datatype), data=sad_inp_ex)
        #         np.savez_compressed("data/cors_{}_{}.npz".format(self.dataset, datatype), data=cors)
        #         np.savez_compressed("data/cors_g_{}_{}.npz".format(self.dataset, datatype), data=cors_g)
        #         np.savez_compressed("data/y_{}_{}.npz".format(self.dataset, datatype), data=y)

        # 进行数据的shuffle和划分
        # 分成了三个部分，将npz中的train分成三部分
        if self.pre_shuffle:
            inp_shape = dae_inp_g.shape[0]
            # train_size = int(inp_shape * 0.7)   # train、val、test数据721
            # val_size = int(inp_shape * 0.8)
            # 进行随机排序，给出一个不确定的顺序 相当于进行了shuffle， 统一进行了shuffle，对于所有用到的array
            random_index = np.random.permutation(inp_shape)

            # 前一个参数是被split的数组，后面的一个参数是将这个数组拆分成两部分82等分
            dae_inp_g = np.split(dae_inp_g[random_index, ...], (inp_shape,))[0]
            dae_inp = np.split(dae_inp[random_index, ...], (inp_shape,))[0]
            dae_inp_ex = np.split(dae_inp_ex[random_index, ...], (inp_shape,))[0]

            sad_inp = np.split(sad_inp[random_index, ...], (inp_shape,))[0]
            sad_inp_ex = np.split(sad_inp_ex[random_index, ...], (inp_shape,))[0]

            cors = np.split(cors[random_index, ...], (inp_shape,))[0]
            cors_g = np.split(cors_g[random_index, ...], (inp_shape,))[0]

            y = np.split(y[random_index, ...], (inp_shape,))[0]

        return dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, y
