# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------


import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from blocks import EqualLinear
from ipdb import set_trace


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., lr_mlp=0.01,use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        # self.use_relu = use_relu

        self.linear = EqualLinear(in_size,out_size,lr_mul=lr_mlp,activation='fused_lrelu')

        # if use_relu:
        #     self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        if len(x.shape)==3:
            shape = x.shape
            x = x.reshape(shape[0]*shape[1],-1)
            x = self.linear(x)
            if self.dropout_r > 0:
                x = self.dropout(x)
            x = x.reshape(shape[0],shape[1],-1)
        else:
            x = self.linear(x)
            if self.dropout_r > 0:
                x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., lr_mlp=0.01, use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r,lr_mlp=lr_mlp, use_relu=use_relu)
        self.linear = EqualLinear(mid_size,out_size,lr_mul=lr_mlp,activation=None)
    def forward(self, x):

        x = self.fc(x)

        if len(x.shape)==3:
            shape = x.shape
            x = x.reshape(shape[0]*shape[1],-1)
            x = self.linear(x)
            x = x.reshape(shape[0],shape[1],-1)
        else:
            x = self.linear(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, cfg):
        super(MHAtt, self).__init__()
        self.cfg = cfg

        self.linear_v = nn.Linear(cfg.NET.HIDDEN_SIZE, cfg.NET.HIDDEN_SIZE)
        self.linear_k = nn.Linear(cfg.NET.HIDDEN_SIZE, cfg.NET.HIDDEN_SIZE)
        self.linear_q = nn.Linear(cfg.NET.HIDDEN_SIZE, cfg.NET.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(cfg.NET.HIDDEN_SIZE, cfg.NET.HIDDEN_SIZE)

        self.dropout = nn.Dropout(cfg.NET.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.cfg.NET.MULTI_HEAD,
            int(self.cfg.NET.HIDDEN_SIZE / self.cfg.NET.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.cfg.NET.MULTI_HEAD,
            int(self.cfg.NET.HIDDEN_SIZE / self.cfg.NET.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.cfg.NET.MULTI_HEAD,
            int(self.cfg.NET.HIDDEN_SIZE / self.cfg.NET.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.cfg.NET.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, cfg):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=cfg.NET.HIDDEN_SIZE,
            mid_size=cfg.NET.FF_SIZE,
            out_size=cfg.NET.HIDDEN_SIZE,
            dropout_r=cfg.NET.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, cfg):
        super(SA, self).__init__()

        self.mhatt = MHAtt(cfg)
        self.ffn = FFN(cfg)

        self.dropout1 = nn.Dropout(cfg.NET.DROPOUT_R)
        self.norm1 = LayerNorm(cfg.NET.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(cfg.NET.DROPOUT_R)
        self.norm2 = LayerNorm(cfg.NET.HIDDEN_SIZE)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, cfg):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(cfg)
        self.mhatt2 = MHAtt(cfg)
        self.ffn = FFN(cfg)

        self.dropout1 = nn.Dropout(cfg.NET.DROPOUT_R)
        self.norm1 = LayerNorm(cfg.NET.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(cfg.NET.DROPOUT_R)
        self.norm2 = LayerNorm(cfg.NET.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(cfg.NET.DROPOUT_R)
        self.norm3 = LayerNorm(cfg.NET.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, cfg):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(cfg) for _ in range(cfg.NET.LAYER)])
        self.dec_list = nn.ModuleList([SGA(cfg) for _ in range(cfg.NET.LAYER)])

    def forward(self, y, x, y_mask, x_mask):
        # y: words_emb bs x nef x seq_len
        # x: grid_feat bs x nef x 64
        y = y.transpose(1, 2)  # bs x seq_len, nef
        x = x.transpose(1, 2)  # bs x 64 x nef 

        # Get encoder last hidden vector
        for enc in self.enc_list:
            y = enc(y, y_mask)  # bs x seq_len x hid_size

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        if x is not None:
            for dec in self.dec_list:
                x = dec(x, y, x_mask, y_mask)
                # bs x 64 x hid_size
        
        return y.transpose(1, 2), x.transpose(1, 2)


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, cfg):
        super(AttFlat, self).__init__()
        self.cfg = cfg

        self.mlp = MLP(
            in_size=cfg['NET.HIDDEN_SIZE'],
            mid_size=cfg['NET.FLAT_MLP_SIZE'],
            out_size=cfg['NET.FLAT_GLIMPSES'],
            lr_mlp=cfg['lr_mlp'],
            dropout_r=cfg['NET.DROPOUT_R'],
            use_relu=True
        )

        self.linear_merge = EqualLinear(
            cfg['NET.HIDDEN_SIZE'] * cfg['NET.FLAT_GLIMPSES'],
            cfg['NET.FLAT_OUT_SIZE'],
            lr_mul=cfg['lr_mlp'],
            activation='fused_lrelu')

    def forward(self, x, x_mask):
        # x: B,L,dim
        att = self.mlp(x)
        if x_mask:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        # att: B,L,dim
        att = F.softmax(att, dim=1)  # softmax on dimension of L

        att_list = []
        for i in range(self.cfg['NET.FLAT_GLIMPSES']):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        # x_atted: B,dim
        x_atted = torch.cat(att_list, dim=1)
        x = x_atted
        if len(x.shape)==3:
            shape = x.shape
            x = x.reshape(shape[0]*shape[1],-1)
            x = self.linear_merge(x)
            x = x.reshape(shape[0],shape[1],-1)
        else:
            x = self.linear_merge(x_atted)
        return x


    