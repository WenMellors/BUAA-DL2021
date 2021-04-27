from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
import torch.nn as nn

from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel
from trafficdl.model.traffic_flow_prediction.data_parameters import data_parameters
from trafficdl.model.loss import masked_rmse_torch

act = 'relu'


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def spatial_posenc(position_r, position_c, d_model):
    angle_rads_r = get_angles(
        position_r, np.arange(d_model)[
            np.newaxis, :], d_model)

    angle_rads_c = get_angles(
        position_c, np.arange(d_model)[
            np.newaxis, :], d_model)

    pos_encoding = np.zeros(angle_rads_r.shape, dtype=np.float32)

    pos_encoding[:, 0::2] = np.sin(angle_rads_r[:, 0::2])

    pos_encoding[:, 1::2] = np.cos(angle_rads_c[:, 1::2])

    return torch.tensor(
        pos_encoding[np.newaxis, ...], dtype=torch.float32).cuda()


class Convs(nn.Module):
    def __init__(self, n_layer, n_filter, l_hist, r_d=0.1):
        super(Convs, self).__init__()

        self.n_layer = n_layer
        self.l_hist = l_hist

        assert n_layer % 3 == 0

        self.convs = nn.ModuleList(
            [nn.ModuleList([nn.Conv2d(in_channels=4, out_channels=n_filter, kernel_size=(3, 3), padding=(1, 1))
                            for _ in range(l_hist)]) for _ in range(n_layer // 3)])
        self.convs += nn.ModuleList(
            [nn.ModuleList([nn.Conv2d(in_channels=64, out_channels=n_filter, kernel_size=(3, 3), padding=(1, 1))
                            for _ in range(l_hist)]) for _ in range(2 * (n_layer // 3))])
        self.dropouts = nn.ModuleList([nn.ModuleList(
            [nn.Dropout(r_d) for _ in range(l_hist)]) for _ in range(n_layer)])

    def forward(self, inps, training):
        outputs = list(torch.split(inps, 1, dim=1))  # TODO

        for i in range(self.n_layer):
            for j in range(self.l_hist):
                if i == 0:
                    outputs[j] = outputs[j].squeeze(1)
                if outputs[j].shape[1] == 7:
                    outputs[j] = self.convs[i][j](
                        outputs[j].permute([0, 2, 1, 3]))
                elif outputs[j].shape[1] == 11:
                    outputs[j] = self.convs[i][j](
                        outputs[j].permute([0, 3, 1, 2]))
                else:
                    outputs[j] = self.convs[i][j](outputs[j])

                outputs[j] = torch.relu(outputs[j])
                outputs[j] = self.dropouts[i][j](outputs[j])
                if i == self.n_layer - 1:
                    outputs[j] = outputs[j].unsqueeze(1)

        output = torch.stack(outputs, dim=1)

        return output


def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = torch.matmul(q, k.transpose(-1, -2))

    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MultiSpaceAttention(nn.Module):
    def __init__(self, d_model, n_head, self_att=True, inplane=64):
        super(MultiSpaceAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.self_att = self_att

        assert d_model % n_head == 0

        self.depth = d_model // n_head

        if self_att:
            self.wx = nn.Linear(inplane, d_model * 3)
        else:
            self.wq = nn.Linear(inplane, d_model)
            self.wkv = nn.Linear(inplane, d_model * 2)

        self.wo = nn.Linear(inplane, d_model)  # TODO

    def split_heads(self, x):
        shape = x.shape
        x = torch.reshape(
            x,
            (shape[0],
             shape[1],
                shape[2],
                self.n_head,
                self.depth))
        return x.permute([0, 1, 3, 2, 4])

    def forward(self, v, k, q, mask):
        if self.self_att:
            wx_o = self.wx(q)
            q, k, v = torch.split(wx_o, wx_o.shape[-1] // 3, dim=-1)
        else:
            q = self.wq(q)
            wkv_o = self.wkv(k)
            k, v = torch.split(wkv_o, wkv_o.shape[-1] // 2, dim=-1)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = scaled_attention.permute([0, 1, 3, 2, 4])

        d_shape = scaled_attention.shape

        concat_attention = torch.reshape(
            scaled_attention, (d_shape[0], d_shape[1], d_shape[2], self.d_model))

        output = self.wo(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff, inplane):
    return nn.Sequential(
        nn.Linear(inplane, dff),
        nn.ReLU(),
        nn.Linear(dff, d_model)
    )


def ex_encoding(d_model, dff, inplane):
    return nn.Sequential(
        nn.Linear(inplane, dff),  # TODO activate func?
        nn.ReLU(),
        nn.Linear(dff, d_model),
        nn.Sigmoid()
    )


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dpo_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.msa = MultiSpaceAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff, 64)

        norm_shape = [64]
        self.layernorm1 = nn.LayerNorm(normalized_shape=norm_shape, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=norm_shape, eps=1e-6)

        self.dropout1 = nn.Dropout(dpo_rate)
        self.dropout2 = nn.Dropout(dpo_rate)

    def forward(self, x, training, mask):
        attn_output, _ = self.msa(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dff, r_d=0.1, revert_q=False):
        super(DecoderLayer, self).__init__()

        self.revert_q = revert_q

        self.msa1 = MultiSpaceAttention(d_model, n_head)
        self.msa2 = MultiSpaceAttention(d_model, n_head, self_att=False)

        self.ffn = point_wise_feed_forward_network(d_model, dff, 64)

        norm_shape = [64]
        self.layernorm1 = nn.LayerNorm(normalized_shape=norm_shape, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=norm_shape, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(normalized_shape=norm_shape, eps=1e-6)

        self.dropout1 = nn.Dropout(r_d)
        self.dropout2 = nn.Dropout(r_d)
        self.dropout3 = nn.Dropout(r_d)

    def forward(self, x, kv, training, look_ahead_mask, threshold_mask):
        attn1, attn_weights_block1 = self.msa1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        if self.revert_q:
            out1_r = out1.permute([0, 2, 1, 3])
            attn2, attn_weights_block2 = self.msa2(
                kv, kv, out1_r, threshold_mask)
            attn2 = attn2.permute([0, 2, 1, 3])
        else:
            attn2, attn_weights_block2 = self.msa2(
                kv, kv, out1, threshold_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class DAE(nn.Module):
    def __init__(
            self,
            n_layer,
            d_model,
            n_head,
            dff,
            conv_layer,
            conv_filter,
            l_hist,
            r_d=0.1):
        super(DAE, self).__init__()

        self.d_model = d_model
        self.n_layer = n_layer

        self.convs = Convs(conv_layer, conv_filter, l_hist, r_d)
        self.convs_g = Convs(conv_layer, conv_filter, l_hist, r_d)

        self.ex_encoder = ex_encoding(d_model, dff, 65)
        self.ex_encoder_g = ex_encoding(d_model, dff, 65)

        self.dropout = nn.Dropout(r_d)
        self.dropout_g = nn.Dropout(r_d)

        self.enc_g = nn.ModuleList(
            [EncoderLayer(d_model, n_head, dff, r_d) for _ in range(n_layer)])
        self.enc_l = nn.ModuleList(
            [DecoderLayer(d_model, n_head, dff, r_d) for _ in range(n_layer)])

    def forward(
            self,
            x,
            x_g,
            ex,
            cors,
            cors_g,
            training,
            threshold_mask,
            threshold_mask_g):
        attention_weights = {}

        shape = x.shape

        ex_enc = self.ex_encoder(ex).unsqueeze(2)
        ex_enc_g = self.ex_encoder_g(ex).unsqueeze(2)

        pos_enc = cors.unsqueeze(1)
        pos_enc_g = cors_g.unsqueeze(1)

        x = self.convs(x.transpose(-2, -1), training)
        x_g = self.convs_g(x_g, training)

        x *= np.sqrt(self.d_model)
        x_g *= np.sqrt(self.d_model)

        x = x.reshape([shape[0], shape[1], -1, self.d_model])
        x_g = x_g.reshape([shape[0], shape[1], -1, self.d_model])

        x = x + ex_enc + pos_enc
        x_g = x_g + ex_enc_g + pos_enc_g

        x = self.dropout(x)
        x_g = self.dropout_g(x_g)

        for i in range(self.n_layer):
            x_g = self.enc_g[i](x_g, training, threshold_mask_g)

        for i in range(self.n_layer):
            x, block1, block2 = self.enc_l[i](
                x, x_g, training, threshold_mask, threshold_mask_g)
            attention_weights['dae_layer{}_block1'.format(i + 1)] = block1
            attention_weights['dae_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights


class SAD(nn.Module):
    def __init__(self, n_layer, d_model, n_head, dff, conv_layer, r_d=0.1):
        super(SAD, self).__init__()

        self.d_model = d_model
        self.n_layer = n_layer
        self.pos_enc = spatial_posenc(0, 0, self.d_model)

        self.ex_encoder = ex_encoding(d_model, dff, 65)
        self.dropout = nn.Dropout(r_d)

        self.li_conv = nn.Sequential()
        self.li_conv.add_module("linear", nn.Linear(2, d_model))
        self.li_conv.add_module("activation_relu", nn.ReLU())
        for i in range(conv_layer - 1):
            self.li_conv.add_module(
                "linear{}".format(i), nn.Linear(
                    d_model, d_model))
            self.li_conv.add_module("activation_relu{}".format(i), nn.ReLU())

        self.dec_s = nn.ModuleList(
            [DecoderLayer(d_model, n_head, dff, r_d) for _ in range(n_layer)])
        self.dec_t = nn.ModuleList(
            [DecoderLayer(d_model, n_head, dff, r_d, revert_q=True) for _ in range(n_layer)])

    def forward(
            self,
            x,
            ex,
            dae_output,
            training,
            look_ahead_mask,
            threshold_mask):
        attention_weights = {}
        ex_enc = self.ex_encoder(ex)

        x = self.li_conv(x)
        x *= np.sqrt(self.d_model)
        x = x + ex_enc + self.pos_enc

        x = self.dropout(x)
        x_s = x.unsqueeze(1)
        x_t = x.unsqueeze(1)

        for i in range(self.n_layer):
            x_s, block1, block2 = self.dec_s[i](
                x_s, dae_output, training, look_ahead_mask, None)
            attention_weights['sad_s_layer{}_block1'.format(i + 1)] = block1
            attention_weights['sad_s_layer{}_block2'.format(i + 1)] = block2

        x_s = x_s.permute([0, 2, 1, 3])

        for i in range(self.n_layer):
            x_t, block1, block2 = self.dec_t[i](
                x_t, x_s, training, look_ahead_mask, None)
            attention_weights['decoder_t_layer{}_block1'.format(
                i + 1)] = block1
            attention_weights['decoder_t_layer{}_block2'.format(
                i + 1)] = block2

        output = x_t.squeeze(1)

        return output, attention_weights


class DsanUse(nn.Module):
    def __init__(
            self,
            n_layer,
            d_model,
            n_head,
            dff,
            conv_layer,
            conv_filter,
            l_hist,
            r_d=0.1):
        super(DsanUse, self).__init__()

        self.dae = DAE(
            n_layer,
            d_model,
            n_head,
            dff,
            conv_layer,
            conv_filter,
            l_hist,
            r_d)

        self.sad = SAD(n_layer, d_model, n_head, dff, conv_layer, r_d)

        self.final_layer = nn.Linear(64, 2)

    def forward(
            self,
            dae_inp_g,
            dae_inp,
            dae_inp_ex,
            sad_inp,
            sad_inp_ex,
            cors,
            cors_g,
            training,
            threshold_mask,
            threshold_mask_g,
            look_ahead_mask):
        dae_output, attention_weights_dae = self.dae(
            dae_inp, dae_inp_g, dae_inp_ex, cors, cors_g, training, threshold_mask, threshold_mask_g)

        sad_output, attention_weights_sad = self.sad(
            sad_inp, sad_inp_ex, dae_output, training, look_ahead_mask, None)

        final_output = self.final_layer(sad_output)
        final_output = torch.tanh(final_output)

        return final_output, attention_weights_dae, attention_weights_sad


class DSAN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature, dataset_name):
        n_layer = config.get('n_layer', 3)
        d_model = config.get('d_model', 64)
        n_head = config.get('n_head', 8)
        dff = 4 * d_model
        conv_layer = config.get('conv_layer', 3)
        conv_filter = config.get('conv_filter', 64)
        n_w = config.get('n_w', 1)
        n_d = config.get('n_d', 3)
        n_wd_times = config.get('n_wd_times', 1)
        n_p = config.get('n_p', 1)
        l_hist = (n_w + n_d) * n_wd_times + n_p

        super(DSAN, self).__init__(config, data_feature)
        self.dsan = DsanUse(
            n_layer,
            d_model,
            n_head,
            dff,
            conv_layer,
            conv_filter,
            l_hist)

        param = data_parameters[dataset_name]
        self.param = param

    def generate_x(self, batch):
        return batch['dae_inp_g'], batch['dae_inp'], batch['dae_inp_ex'], batch[
            'sad_inp'], batch['sad_inp_ex'], batch['cors'], batch['cors_g'], batch['y']

    # , batch['threshold_mask_g'], batch['threshold_mask'], batch['combined_mask']\
    def predict(self, batch):
        param = self.param
        pred_type = param['pred_type']
        dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, \
            y = self.generate_x(batch)
        # threshold_mask_g, threshold_mask, combined_mask,

        # 这三个部分在Mydataset里面已经处理过了
        threshold_mask_g, threshold_mask, combined_mask = create_masks(
            dae_inp_g[..., :pred_type], dae_inp[..., :pred_type], sad_inp)

        res, _, _ = self.dsan(dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex,
                              cors, cors_g, True, threshold_mask, threshold_mask_g, combined_mask)
        return res

    def calculate_loss(self, batch):
        param = self.param
        pred_type = param['pred_type']
        dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, y = self.generate_x(
            batch)

        # 这三个部分在Mydataset里面已经处理过了
        threshold_mask_g, threshold_mask, combined_mask = create_masks(
            dae_inp_g[..., :pred_type], dae_inp[..., :pred_type], sad_inp)

        res, _, _ = self.dsan(dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex,
                              cors, cors_g, True, threshold_mask, threshold_mask_g, combined_mask)

        loss = masked_rmse_torch(res, y)  # 这里的这  个是nn.model里面自带的函数吗
        return loss


def create_look_ahead_mask(size):
    mask = 1 - torch.tril(torch.ones((size, size)), diagonal=-1)
    return mask.cuda()


def create_threshold_mask(inp):
    oup = torch.sum(inp, dim=-1)
    shape = oup.shape
    oup = torch.reshape(oup, [shape[0], shape[1], -1])
    mask = (oup == 0).float()
    return mask


def create_threshold_mask_tar(inp):
    oup = torch.sum(inp, dim=-1)
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
