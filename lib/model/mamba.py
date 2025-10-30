import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

from mamba_ssm import Mamba
import copy

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# -------------------------------bridge --------------------
class Scale_reduce(nn.Module):
    def __init__(self, dim, reduction_ratio):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio

        self.sr0 = nn.Conv2d(dim, dim, reduction_ratio[3], reduction_ratio[3])
        self.sr1 = nn.Conv2d(dim * 2, dim * 2, reduction_ratio[2], reduction_ratio[2])
        self.sr2 = nn.Conv2d(dim * 4, dim * 4, reduction_ratio[1], reduction_ratio[1])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = W = int(math.sqrt(N // 120))

        tem0 = x[:, :64 * H * W, :].reshape(B, H * 8, W * 8, C).permute(0, 3, 1, 2)
        tem1 = x[:, 64 * H * W:96 * H * W, :].reshape(B, H * 4, W * 4, C * 2).permute(0, 3, 1, 2)
        tem2 = x[:, 96 * H * W:112 * H * W, :].reshape(B, H * 2, W * 2, C * 4).permute(0, 3, 1, 2)
        tem3 = x[:, 112 * H * W:120 * H * W, :]

        sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
        sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
        sr_2 = self.sr2(tem2).reshape(B, C, -1).permute(0, 2, 1)

        reduce_out = self.norm(torch.cat([sr_0, sr_1, sr_2, tem3], -2))

        return reduce_out


class M_EfficientChannelAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio  # list[1  2  4  8]
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"\n inside channel att")
        B, N, C = x.shape
        k = self.k(x).reshape((B, C, N))
        q = self.q(x).reshape((B, C, N))
        v = self.v(x).reshape((B, C, N))

        # q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        head_k_ch = C // self.head
        head_v_ch = C // self.head

        attended_values = []
        for i in range(self.head):
            key = F.softmax(k[
                            :,
                            i * head_k_ch: (i + 1) * head_k_ch,
                            :
                            ], dim=2)

            query = F.softmax(q[
                              :,
                              i * head_k_ch: (i + 1) * head_k_ch,
                              :
                              ], dim=1)

            value = v[
                    :,
                    i * head_v_ch: (i + 1) * head_v_ch,
                    :
                    ]

            context = key @ value.transpose(1, 2)  # dk*dv
            # print(f'context:{context.shape}')
            attended_value = (context.transpose(1, 2) @ query)
            # print(f'attended_value:{attended_value.shape}')
            attended_value = attended_value.reshape(B, head_v_ch, N)  # n*dv
            # print(f'reshaped attended_value:{attended_value.shape}')
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        # print(f'aggregated_values: {aggregated_values.shape}')
        out = self.proj(aggregated_values.permute((0, 2, 1)))
        # print(f'out of attention: {out.shape}')

        return out


class M_EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio  # list[1  2  4  8]
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio is not None:
            self.scale_reduce = Scale_reduce(dim, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio is not None:
            x = self.scale_reduce(x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out


class M_MambaAtten(nn.Module):
    def __init__(self, dim=96, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba_forw = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            use_fast_path=False,
        )
        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)
        self.mamba_backw = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            use_fast_path=False,

        )
        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.flip(x, dims=[1])
        x_mamba = self.mamba_forw(x)
        y_mamba = self.mamba_backw(y)
        a = torch.flip(y_mamba, dims=[1])

        x_out = self.out_proj(x_mamba + torch.flip(y_mamba, dims=[1]))
        return x + x_out


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        # print('input in DWConv: {}'.format(x.shape))
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class BridgLayer_para(nn.Module):
    def __init__(self, dims, head, reduction_ratios, att):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        if att == "CA":
            self.attn = M_EfficientChannelAtten(dims, head, reduction_ratios)
        elif att == "SA":
            self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        elif att == "Mamba":
            self.attn = M_MambaAtten(dims)
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = ConvolutionalGLU(in_features=dims, hidden_features=dims * 4)
        self.mixffn2 = ConvolutionalGLU(in_features=dims * 2, hidden_features=dims * 8)
        self.mixffn3 = ConvolutionalGLU(in_features=dims * 4, hidden_features=dims * 16)
        self.mixffn4 = ConvolutionalGLU(in_features=dims * 8, hidden_features=dims * 32)

    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 96
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3, c4 = inputs
            B, C, H1, W1 = c1.shape
            H, W = H1 // 8, W1 // 8
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64

            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)
        else:
            B, N, C = inputs.shape
            H = W = int(math.sqrt(N // 120))

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        tem1 = tx[:, :64 * H * W, :].reshape(B, -1, C)
        tem2 = tx[:, 64 * H * W:96 * H * W, :].reshape(B, -1, C * 2)
        tem3 = tx[:, 96 * H * W:112 * H * W, :].reshape(B, -1, C * 4)
        tem4 = tx[:, 112 * H * W:120 * H * W, :].reshape(B, -1, C * 8)

        m1f = self.mixffn1(tem1, H * 8, W * 8).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, H * 4, W * 4).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, H * 2, W * 2).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, H, W).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f, m3f, m4f], -2)

        tx2 = tx1 + t1
        return tx2


class BridgeBlock_para(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()

        self.bridge_layer1 = BridgLayer_para(dims, head, reduction_ratios, "Mamba")  # channel
        self.bridge_layer2 = BridgLayer_para(dims, head, reduction_ratios, "CA")  # spatial
        self.bridge_layer3 = BridgLayer_para(dims, head, reduction_ratios, "SA")
        self.bridge_layer4 = BridgLayer_para(dims, head, reduction_ratios, "Mamba")
        self.proj_act = nn.Sequential(
            nn.Linear(2 * dims, dims),
            nn.LayerNorm(dims),
            nn.GELU()
        )
    def forward(self, x):
        bridge1 = self.bridge_layer1(x)  # [B N C]
        bridge2 = self.bridge_layer2(bridge1)  # [B N C]
        bridge3 = self.bridge_layer3(bridge1)
        B, N, C = bridge1.shape
        H = W = int(math.sqrt(N // 120))
        bridge4 = torch.cat((bridge2, bridge3), dim=2)  # [B N 2C]
        bridge4 = self.proj_act(bridge4)
        br_dual = self.bridge_layer4(bridge4)
        outs = []

        sk1 = br_dual[:, :64 * H * W, :].reshape(B, H * 8, W * 8, C).permute(0, 3, 1, 2)
        sk2 = br_dual[:, 64 * H * W:96 * H * W, :].reshape(B, H * 4, W * 4, C * 2).permute(0, 3, 1, 2)
        sk3 = br_dual[:, 96 * H * W:112 * H * W, :].reshape(B, H * 2, W * 2, C * 4).permute(0, 3, 1, 2)
        sk4 = br_dual[:, 112 * H * W:120 * H * W, :].reshape(B, H, W, C * 8).permute(0, 3, 1, 2)
        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)
        return outs


# ----------------------------------------------------------
# ---------------------------edge-aware --------------------------------
class ConvBNR(nn.Module):
    def __init__(self, input_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(input_channel, out_channel, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class GateFusion(nn.Module):
    def __init__(self, in_planes):
        self.init__ = super(GateFusion, self).__init__()

        self.gate_1 = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)
        self.gate_2 = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        ###
        cat_fea = torch.cat([x1, x2], dim=1)

        ###
        att_vec_1 = self.gate_1(cat_fea)
        att_vec_2 = self.gate_2(cat_fea)

        att_vec_cat = torch.cat([att_vec_1, att_vec_2], dim=1)
        att_vec_soft = self.softmax(att_vec_cat)

        att_soft_1, att_soft_2 = att_vec_soft[:, 0:1, :, :], att_vec_soft[:, 1:2, :, :]
        x_fusion = x1 * att_soft_1 + x2 * att_soft_2

        return x_fusion


class SobelConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, requires_grad=True):
        assert kernel_size % 2 == 1, 'SobelConv2d\'s kernel_size must be odd.'
        assert out_channels % 4 == 0, 'SobelConv2d\'s out_channels must be a multiple of 4.'
        assert out_channels % groups == 0, 'SobelConv2d\'s out_channels must be a multiple of groups.'

        super(SobelConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        # Initialize the Sobel kernal
        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 4 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 4 == 2:
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            else:
                self.sobel_weight[idx, :, -1, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, 0, -1] = 2

        # Define the trainable sobel factor
        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)

    def forward(self, x):
        if torch.cuda.is_available():
            self.sobel_factor = self.sobel_factor.cuda()
            if isinstance(self.bias, nn.Parameter):
                self.bias = self.bias.cuda()

        sobel_weight = self.sobel_weight * self.sobel_factor

        if torch.cuda.is_available():
            sobel_weight = sobel_weight.cuda()

        out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out


class EAM(nn.Module):
    def __init__(self, in_ch=3, sobel_ch=48):
        super(EAM, self).__init__()
        self.layer0 = ConvBNR(input_channel=96, out_channel=96, kernel_size=1, padding=0)
        self.layer1 = ConvBNR(input_channel=192, out_channel=96, kernel_size=1, padding=0)
        self.layer2 = ConvBNR(input_channel=96, out_channel=48, kernel_size=3, padding=1)
        self.low_fusion = GateFusion(96)
        self.sober = SobelConv2d(in_ch, sobel_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.final = nn.Conv2d(96, 1, 1)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x, x1, x2):
        sober_x = self.sober(x)
        x1 = self.layer0(x1)
        x2 = self.up_2(x2)
        x2 = self.layer1(x2)
        fusion_x = self.low_fusion(x1, x2)
        fusion_x = self.up_4(fusion_x)
        fusion_x = self.layer2(fusion_x)
        x_out = self.final(torch.cat((fusion_x, sober_x), dim=1))
        return x_out


# ------------------------------------------------------------------------
# --------------------------------BSA--------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_out)
        return self.sigmoid(x_out) * x

class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3,
        bias=True, bn=False, act=nn.ReLU(True)):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(in_channels=n_feat, out_channels=n_feat, padding=1, kernel_size=kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(ChannelAttention(n_feat))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class Conv2d_batchnorm(torch.nn.Module):
    """
    2D Convolutional layers
    """

    def __init__(
            self,
            num_in_filters,
            num_out_filters,
            kernel_size,
            stride=(1, 1),
            activation="LeakyReLU",
    ):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        return self.activation(x)


class Fuse_Emd(nn.Module):
    def __init__(self, feat_size=None):
        super(Fuse_Emd, self).__init__()
        if feat_size is None:
            feat_size = [96, 192, 384, 768]
        self.feat_size = feat_size
        self.fuse_filters = self.feat_size[0] + self.feat_size[1] + self.feat_size[2] + self.feat_size[3]
        self.cnv_blks1 = Conv2d_batchnorm(self.fuse_filters, self.feat_size[3], 3)
        self.cnv_blks2 = Conv2d_batchnorm(self.fuse_filters, self.feat_size[2], 3)
        self.cnv_blks3 = Conv2d_batchnorm(self.fuse_filters, self.feat_size[1], 3)
        self.cnv_blks4 = Conv2d_batchnorm(self.fuse_filters, self.feat_size[0], 3)
        self.no_param_up = torch.nn.Upsample(scale_factor=2)  # used for upsampling
        self.no_param_down = torch.nn.AvgPool2d(2)  # used for downsampling
        self.sa = SpatialAttention()

    def forward(self, x1, x2, x3, x4):
        x_4 = x4 + self.cnv_blks1(
            torch.cat(
                [x4,
                 self.no_param_down(x3),
                 self.no_param_down(self.no_param_down(x2)),
                 self.no_param_down(self.no_param_down(self.no_param_down(x1))),
                 ],
                dim=1,
            )
        )
        x_4 = x_4 + self.sa(x_4)

        x_3 = x3 + self.cnv_blks2(
            torch.cat(
                [self.no_param_up(x4),
                 x3,
                 self.no_param_down(x2),
                 self.no_param_down(self.no_param_down(x1)),
                 ],
                dim=1,
            )
        )
        x_3 = x_3 + self.sa(x_3)


        x_2 = x2 + self.cnv_blks3(
            torch.cat(
                [self.no_param_up(self.no_param_up(x4)),
                 self.no_param_up(x3),
                 x2,
                 self.no_param_down(x1),
                 ],
                dim=1,
            )
        )
        x_2 = x_2 + self.sa(x_2)


        x_1 = x1 + self.cnv_blks4(
            torch.cat(
                [self.no_param_up(self.no_param_up(self.no_param_up(x4))),
                 self.no_param_up(self.no_param_up(x3)),
                 self.no_param_up(x2),
                 x1,
                 ],
                dim=1,
            )
        )
        x_1 = x_1 + self.sa(x_1)

        return x_1, x_2, x_3, x_4


class BSA(nn.Module):
    def __init__(self, channel, edge_channel=1):
        super(BSA, self).__init__()
        self.rca_x = RCAB(channel)
        self.rca_rx = RCAB(channel)
        self.conv_f = nn.Conv2d(2 * channel + edge_channel, channel, kernel_size=1, stride=1, padding=0)
    def forward(self, x, fuse, edg):
        edg = F.upsample(edg, size=x.shape[2], mode='bilinear', align_corners=False)
        r_x = -1 * (torch.sigmoid(x)) + 1
        x_1 = fuse * x + fuse
        x_1 = self.rca_x(x_1)
        r_x_1 = fuse * r_x + fuse
        r_x_1 = self.rca_rx(r_x_1)
        x_2 = x_1 * edg + x_1
        r_x_2 = r_x_1 * edg + r_x_1
        out = self.conv_f(torch.concat([x_2, r_x_2, edg], dim=1))
        return out


# -------------------------------------------------------------------------
class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSMEncoder(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, depths=[2, 2, 9, 2],
                 dims=[96, 192, 384, 768], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        self.ape = False
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsamples.append(PatchMerging2D(dim=dims[i_layer], norm_layer=norm_layer))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x_ret = []

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for s, layer in enumerate(self.layers):
            x = layer(x)
            x_ret.append(x.permute(0, 3, 1, 2))
            if s < len(self.downsamples):
                x = self.downsamples[s](x)

        return x_ret


class EMamba(nn.Module):
    def __init__(
            self,
            in_chans=3,
            out_chans=1,
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name="instance",
            res_block: bool = True,
            spatial_dims=2,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, feat_size[0], kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(feat_size[0], eps=1e-5, affine=True),
        )
        self.spatial_dims = spatial_dims
        self.vssm_encoder = VSSMEncoder(patch_size=2, in_chans=feat_size[0])
        self.res_block = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        # deep supervision support
        self.out_layers = nn.ModuleList()
        for i in range(4):
            self.out_layers.append(UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[i],
                out_channels=self.out_chans
            ))

        self.bridge = BridgeBlock_para(dims=96, head=1, reduction_ratios=[1, 2, 4, 8])
        self.edge = EAM()
        self.bsa1 = BSA(channel=768)
        self.bsa2 = BSA(channel=384)
        self.bsa3 = BSA(channel=192)
        self.bsa4 = BSA(channel=96)

        self.fuse_emd = Fuse_Emd()
    def forward(self, x_in):
        x1 = self.stem(x_in)
        vss_outs = self.vssm_encoder(x1)
        x1 = self.res_block(x1)
        enc1 = self.encoder1(vss_outs[0])
        enc2 = self.encoder2(vss_outs[1])
        enc3 = self.encoder3(vss_outs[2])
        enc4 = self.encoder4(vss_outs[3])
        edge = self.edge(x_in, enc1, enc2)
        enc1, enc2, enc3, enc4 = self.bridge([enc1, enc2, enc3, enc4])
        f1, f2, f3, f4 = self.fuse_emd(enc1, enc2, enc3, enc4)

        dec3 = self.bsa1(enc4, f4, edge)
        dec3 = self.decoder4(dec3, f3)
        # dec3 = dec3 + dec3 * F.upsample(edge, size=dec3.shape[2], mode='bilinear', align_corners=False)

        dec2 = self.bsa2(dec3, f3, edge)
        dec2 = self.decoder3(dec2, f2)
        # dec2 = dec2 + dec2 * F.upsample(edge, size=dec2.shape[2], mode='bilinear', align_corners=False)

        dec1 = self.bsa3(dec2, f2, edge)
        dec1 = self.decoder2(dec1, f1)
        # dec1 = dec1 + dec1 * F.upsample(edge, size=dec1.shape[2], mode='bilinear', align_corners=False)

        dec0 = self.bsa4(dec1, f1, edge)
        dec0 = self.decoder1(dec0, x1)
        # dec0 = dec0 + dec0 * F.upsample(edge, size=dec0.shape[2], mode='bilinear', align_corners=False)
        # dec_out = self.bsa5(dec_out, edge)


        feat_out = [dec0, dec1, dec2, dec3]
        out = []
        for i in range(4):
            pred = self.out_layers[i](feat_out[i])
            pred = F.interpolate(pred, scale_factor=2 ** (i+1), mode='bilinear')
            out.append(pred)

        # edge = F.upsample(edge, size=out[0].shape[2], mode='bilinear', align_corners=False)

        return out, edge

    @torch.no_grad()
    def freeze_encoder(self):
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True


def load_pretrained_ckpt(
        model,
        ckpt_path='/home/zbw/Polyp/lib/weight/vssmsmall_dp03.pth'
):
    print(f"Loading weights from: {ckpt_path}")
    skip_params = ["norm.weight", "norm.bias", "head.weight", "head.bias",
                   "patch_embed.proj.weight", "patch_embed.proj.bias",
                   "patch_embed.norm.weight", "patch_embed.norm.weight"]

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_dict = model.state_dict()
    for k, v in ckpt['model'].items():
        if k in skip_params:
            # print(f"Skipping weights: {k}")
            continue
        kr = f"vssm_encoder.{k}"
        if "downsample" in kr:
            i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
            kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
            assert kr in model_dict.keys()
        if kr in model_dict.keys():
            assert v.shape == model_dict[kr].shape, f"Shape mismatch: {v.shape} vs {model_dict[kr].shape}"
            model_dict[kr] = v
        else:
            # print(f"Passing weights: {k}")
            pass
    model.load_state_dict(model_dict)

    return model


def get_mamba():
    model = EMamba(
        in_chans=3,
        out_chans=1,
        feat_size=[48, 96, 192, 384],
        hidden_size=768,
    )

    model = load_pretrained_ckpt(model)

    return model


if __name__ == '__main__':
    model = get_mamba()
    model = model.cuda()
    input_tensor = torch.randn(1, 3, 384, 384).cuda()

    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from thop import profile
    flops, params = profile(model, (input_tensor,))
    print('flops: ', flops/10000000000, 'params: ', params/1000000)
    #
    # input_tensor2 = torch.randn(1, 3, 256, 256).cuda()
    # input_tensor3 = torch.randn(1, 3, 384, 384).cuda()
    # [d0, d1, d2, d3, d4], edg = model(input_tensor)
    # print(d0.shape, d1.shape, d2.shape, d3.shape, d4.shape, edg.shape)
    # [d0, d1, d2, d3, d4], edg = model(input_tensor2)
    # print(d0.shape, d1.shape, d2.shape, d3.shape, d4.shape, edg.shape)
