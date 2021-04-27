import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(
            self,
            dae_inp_g,
            dae_inp,
            dae_inp_ex,
            sad_inp,
            sad_inp_ex,
            cors,
            cors_g,
            y,
            pred_type):
        """
        data: 必须是一个 list
        """
        self.dae_inp_g = dae_inp_g
        self.dae_inp = dae_inp
        self.dae_inp_ex = dae_inp_ex
        self.sad_inp = sad_inp
        self.sad_inp_ex = sad_inp_ex
        self.cors = cors
        self.cors_g = cors_g

        # self.dae_inp_g = torch.from_numpy(dae_inp_g)
        # self.dae_inp = torch.from_numpy(dae_inp)
        # self.dae_inp_ex = torch.from_numpy(dae_inp_ex)
        # self.sad_inp = torch.from_numpy(sad_inp)
        # self.sad_inp_ex = torch.from_numpy(sad_inp_ex)
        # self.cors = torch.from_numpy(cors)
        # self.cors_g = torch.from_numpy(cors_g)

        # threshold_mask_g, threshold_mask, combined_mask = \
        #     create_masks(self.dae_inp_g[..., :pred_type], self.dae_inp[..., :pred_type], self.sad_inp)
        #
        # self.threshold_mask_g = threshold_mask_g  # 后面的几个mask的变量
        # self.threshold_mask = threshold_mask
        # self.combined_mask = combined_mask

        self.y = y

    def __getitem__(self, index):
        return self.dae_inp_g[index], self.dae_inp[index], self.dae_inp_ex[index], self.sad_inp[index], \
            self.sad_inp_ex[index], self.cors[index], self.cors_g[index], self.y[index]
        # self.threshold_mask_g[index], self.threshold_mask[index], self.combined_mask, self.y[index]

    def __len__(self):
        return self.dae_inp_g.shape[0]


def create_look_ahead_mask(size):
    mask = 1 - torch.tril(torch.ones((size, size)), diagonal=-1)
    return mask


def create_threshold_mask(inp):
    oup = torch.sum(inp, -1)
    shape = oup.shape
    oup = torch.reshape(oup, [shape[0], shape[1], -1])
    mask = (oup == 0).float()
    return mask


def create_threshold_mask_tar(inp):
    oup = torch.sum(inp, -1)
    mask = (oup == 0).float()
    return mask


def create_masks(inp_g, inp_l, tar):
    threshold_mask_g = create_threshold_mask(inp_g).unsqueeze(2).unsqueeze(3)
    threshold_mask = create_threshold_mask(inp_l).unsqueeze(2).unsqueeze(3)

    look_ahead_mask = create_look_ahead_mask(tar.shape[1])
    dec_target_threshold_mask = create_threshold_mask_tar(
        tar).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    combined_mask = torch.max(dec_target_threshold_mask, look_ahead_mask)

    return threshold_mask_g, threshold_mask, combined_mask
