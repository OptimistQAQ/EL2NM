import math
import torch
import torch.nn as nn

import numpy as np

import scipy.io

from pathlib import Path
_script_dir = Path( __file__ ).parent
_root_dir = _script_dir.parent


class ForwardProcessBase():
    
    def forward(self, x, i):
        pass

    @torch.no_grad()
    def reset_parameters(self, batch_size=32):
        pass

class Gauss_Forward(ForwardProcessBase):
    
    def __init__(self,
                 image_size=(16, 3, 128, 128),
                 num_timesteps=50,
                 batchsize=1,
                 random_gauss=False,
                 device='cuda'):
        self.num_timesteps = num_timesteps
        self.batchsize = batchsize
        self.image_size = image_size
        self.random_gauss = random_gauss
        self.dtype = torch.float32
        self.beta = []
        self.device = device
        
        self.shot_noise = torch.tensor(-0.1036, dtype = self.dtype)
        self.read_noise = torch.tensor(-0.0209, dtype = self.dtype)
        self.row_noise = torch.tensor(-0.0010, dtype = self.dtype)
        self.row_noise_temp = torch.tensor(0.0055, dtype = self.dtype)
        self.uniform_noise = torch.tensor(-0.0770, dtype = self.dtype)
        mean_noise = scipy.io.loadmat(str(_root_dir) + '/data/fixed_pattern_noise.mat')['mean_pattern']
        fixed_noise = mean_noise.astype('float32')/2**16
        self.fixednoiset = torch.tensor(fixed_noise.transpose(2,0,1), dtype = self.dtype).unsqueeze(0)
        self.periodic_params = torch.tensor([-0.0015, -0.2673,  0.0021], dtype = self.dtype)*100 #*1000
        
        self.generate_gauss_step()

    @torch.no_grad()
    def reset_parameters(self, batch_size=16):
        if batch_size != -1:
            self.batchsize = batch_size
        if self.random_gauss:
            self.generate_gauss_step()
    
    @torch.no_grad()    
    def generate_gauss_step(self):
        if not self.random_gauss:
            rstate = np.random.get_state()
            np.random.seed(123321)
        variance = []
        variance.append(self.shot_noise)
        variance.append(self.read_noise)
        variance.append(self.uniform_noise)
        variance.append(self.row_noise)
        variance.append(self.row_noise_temp)
        beta_start = 0.0001
        beta_end = 0.02
        
        beta = torch.linspace(beta_start, beta_end, self.num_timesteps).tolist()
        self.beta = beta
        
        self.gauss = []
        
        # t = torch.randint(low=0, high=50, size=(16 // 2 + 1,))
        # t = torch.cat([t, 50 - t - 1], dim=0)[:16]
        
        for i in range(self.num_timesteps):
            # e = (torch.randn((16, 3, 128, 128)) * self.read_noise).to(self.device)
            # a = torch.tensor(1 - beta[i]).to(self.device)
            # self.gauss.append(e * (1 - a).sqrt())
            if i < 41:
                e = (torch.randn((16, 3, 128, 128)) * self.read_noise).to(self.device)
                a = torch.tensor(1 - beta[i]).to(self.device)
                self.gauss.append(e * (1 - a).sqrt())
            elif i == 41:
                ee = torch.randn((16, 3, 128, 128))
                e = (torch.randn(*ee.shape[0:-3], ee.shape[-1]) * self.row_noise_temp).to(self.device).unsqueeze(-2).unsqueeze(-2)
                a = torch.tensor(1 - beta[i]).to(self.device)
                self.gauss.append(e * (1-a).sqrt())
            elif i >= 42 and i < 43:
                ee = torch.randn((16, 3, 128, 128))
                e = (torch.randn([*ee.shape[0:-2],ee.shape[-1]]) * self.row_noise).to(self.device).unsqueeze(-2)
                a = torch.tensor(1 - beta[i]).to(self.device)
                self.gauss.append(e * (1-a).sqrt())
            # elif i == 43:
            #     ee = torch.randn((16, 3, 128, 128))
            #     i1 = np.random.randint(0, self.fixednoiset.shape[-2] - ee.shape[-2])
            #     i2 = np.random.randint(0, self.fixednoiset.shape[-1] - ee.shape[-1])
            #     e = self.fixednoiset[...,i1:i1+ee.shape[-2], i2:i2 + ee.shape[-1]].to(self.device)
            #     self.gauss.append(e[:, :3, :, :])
            else:
                e = (torch.rand((16, 3, 128, 128)) * self.uniform_noise).to(self.device)
                a = torch.tensor(1 - beta[i]).to(self.device)
                self.gauss.append(e * (1 - a).sqrt())

    @torch.no_grad()
    def forward(self, x, i):
        gauss_img = x * torch.tensor(1 - self.beta[i]).sqrt().to(self.device)  + self.gauss[i]
        gauss_img = torch.clip(gauss_img, 0, 1)
        return gauss_img

def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 自注意力
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class DiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels * 2 if config.data.conditional else config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # 下采样
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # 上采样
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # 下采样
        hs = [self.conv_in(x)]
        # self.num_resolutions=4
        for i_level in range(self.num_resolutions):
            # self.num_resolutions=2
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # 上采样
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


# import argparse
# import os
# import yaml
# import onnx

# def config_get():
#     parser = argparse.ArgumentParser()
#     # 参数配置文件路径
#     parser.add_argument("--config", default='../scripts/configs.yml', type=str, required=False, help="Path to the config file")
#     args = parser.parse_args()

#     with open(os.path.join(args.config), "r") as f:
#         config = yaml.safe_load(f)
#     new_config = dict2namespace(config)

#     return new_config


# def dict2namespace(config):
#     namespace = argparse.Namespace()
#     for key, value in config.items():
#         if isinstance(value, dict):
#             new_value = dict2namespace(value)
#         else:
#             new_value = value
#         setattr(namespace, key, new_value)
#     return namespace

# config = config_get()

# net = DiffusionUNet(config=config)
# print(net)
