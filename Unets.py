# adapted from https://github.com/TeaPearce/Conditional_Diffusion_MNIST/blob/main/script.py
import torch
import math
from functools import partial
from torch import nn
from torch.nn import Module, ModuleList

import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class Unet(nn.Module):
    def __init__(self, in_channels, n_feat = 256):
        super(Unet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # embed time step
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1+ temb1, down2)  # add embeddings
        up3 = self.up2(up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

class ConditionalUnet_baseline(nn.Module):
    """M0 / baseline：显式条件（one-hot + 全局乘加注入 cemb*feat + temb）。"""
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ConditionalUnet_baseline, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask=None):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        # context_mask: [B], 1 = drop class (classifier-free); 0 = keep label
        if context_mask is not None:
            c = c * (1.0 - context_mask.float().view(-1, 1))

        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


class SpatialSelfAttention(nn.Module):
    """瓶颈分辨率上的空间自注意力（单处）：HW 展平做 MHSA，残差连接。"""

    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        assert channels % heads == 0, "channels 必须整除 heads"
        self.heads = heads
        self.dim_head = channels // heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        xn = self.norm(x)
        qkv = self.qkv(xn)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(B, self.heads, self.dim_head, H * W).transpose(-1, -2)
        k = k.reshape(B, self.heads, self.dim_head, H * W)
        v = v.reshape(B, self.heads, self.dim_head, H * W).transpose(-1, -2)
        scale = self.dim_head ** -0.5
        attn = torch.softmax((q @ k) * scale, dim=-1)
        out = attn @ v
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        out = self.proj(out)
        return x + out


class ConditionalUnet_baseline_attn(nn.Module):
    """baseline + 在最深特征图（down2 输出，进入全局池化之前）插入一处 SpatialSelfAttention。"""

    def __init__(self, in_channels: int, n_feat: int = 256, n_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.attn = SpatialSelfAttention(2 * n_feat, heads=4)
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask=None):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down2 = self.attn(down2)
        hiddenvec = self.to_vec(down2)

        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        if context_mask is not None:
            c = c * (1.0 - context_mask.float().view(-1, 1))

        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


# 兼容旧 import：ConditionalUnet 即 baseline
ConditionalUnet = ConditionalUnet_baseline


def _gn_groups(channels: int, preferred: int = 8) -> int:
    g = min(preferred, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(g, 1)


class FiLM_AdaGN(nn.Module):
    """FiLM：cond -> gamma, beta；x <- x * (1 + gamma) + beta（用于 MinAdaGN）。"""

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.lin = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], cond: [B, cond_dim]
        gb = self.lin(cond)
        gamma, beta = gb.chunk(2, dim=1)
        return x * (1 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)


class ConditionalUnet_MinAdaGN(nn.Module):
    """
    最小 AdaGN / FiLM：结构与 ConditionalUnet_baseline 相同，仅在两处全局融合后追加 FiLM。
      fused1 = cemb1 * up0(h) + temb1  -> FiLM(fused1, [cemb1;temb1])
      fused2 = cemb2 * up2 + temb2    -> FiLM(fused2, [cemb2;temb2])
    不改变编码器、不改变 UnetUp 内部卷积；参数量仅增加两个 Linear(cond->2C)。
    """

    def __init__(self, in_channels: int, n_feat: int = 256, n_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.cond_dim_high = 4 * n_feat  # concat(cemb1_vec, temb1_vec), each 2*n_feat
        self.cond_dim_low = 2 * n_feat  # concat(cemb2_vec, temb2_vec), each n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

        self.film_after_fuse1 = FiLM_AdaGN(self.cond_dim_high, 2 * n_feat)
        self.film_after_fuse2 = FiLM_AdaGN(self.cond_dim_low, n_feat)

    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, context_mask=None):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        c = F.one_hot(c, num_classes=self.n_classes).type(torch.float)
        if context_mask is not None:
            c = c * (1.0 - context_mask.float().view(-1, 1))

        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        cond_h = torch.cat(
            [cemb1.squeeze(-1).squeeze(-1), temb1.squeeze(-1).squeeze(-1)], dim=1
        )
        cond_l = torch.cat(
            [cemb2.squeeze(-1).squeeze(-1), temb2.squeeze(-1).squeeze(-1)], dim=1
        )

        up1_spatial = self.up0(hiddenvec)
        fused1 = cemb1 * up1_spatial + temb1
        fused1 = self.film_after_fuse1(fused1, cond_h)

        up_mid = self.up1(fused1, down2)
        fused2 = cemb2 * up_mid + temb2
        fused2 = self.film_after_fuse2(fused2, cond_l)

        up3 = self.up2(fused2, down1)
        return self.out(torch.cat((up3, x), 1))


class ConditionalUnet_baseline_64(nn.Module):
    """
    64×64 baseline：三次下采样 64→32→16→8，瓶颈 8×8 / 2*n_feat；
    两次 time/class 注入（与 28×28 相同语义），第三次上采样仅 concat skip。
    """

    def __init__(self, in_channels: int, n_feat: int = 256, n_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.img_size = 64

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, n_feat)
        self.down3 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 8, 8),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, context_mask=None):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        hiddenvec = self.to_vec(down3)

        c = F.one_hot(c, num_classes=self.n_classes).type(torch.float)
        if context_mask is not None:
            c = c * (1.0 - context_mask.float().view(-1, 1))

        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        u0 = self.up0(hiddenvec)
        u1 = self.up1(cemb1 * u0 + temb1, down3)
        u2 = self.up2(cemb2 * u1 + temb2, down2)
        u3 = self.up3(u2, down1)
        return self.out(torch.cat((u3, x), 1))


class ConditionalUnet_baseline_attn_64(nn.Module):
    """64×64 baseline + down3 之后、AdaptiveAvgPool 之前一处 SpatialSelfAttention。"""

    def __init__(self, in_channels: int, n_feat: int = 256, n_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.img_size = 64

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, n_feat)
        self.down3 = UnetDown(n_feat, 2 * n_feat)
        self.attn = SpatialSelfAttention(2 * n_feat, heads=4)
        self.to_vec = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 8, 8),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, context_mask=None):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down3 = self.attn(down3)
        hiddenvec = self.to_vec(down3)

        c = F.one_hot(c, num_classes=self.n_classes).type(torch.float)
        if context_mask is not None:
            c = c * (1.0 - context_mask.float().view(-1, 1))

        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        u0 = self.up0(hiddenvec)
        u1 = self.up1(cemb1 * u0 + temb1, down3)
        u2 = self.up2(cemb2 * u1 + temb2, down2)
        u3 = self.up3(u2, down1)
        return self.out(torch.cat((u3, x), 1))


class ConditionalUnet_MinAdaGN_64(nn.Module):
    """64×64 MinAdaGN：与 baseline_64 同拓扑，两处融合后加 FiLM_AdaGN。"""

    def __init__(self, in_channels: int, n_feat: int = 256, n_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.img_size = 64

        self.cond_dim_high = 4 * n_feat
        self.cond_dim_low = 2 * n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, n_feat)
        self.down3 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 8, 8),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

        self.film_after_fuse1 = FiLM_AdaGN(self.cond_dim_high, 2 * n_feat)
        self.film_after_fuse2 = FiLM_AdaGN(self.cond_dim_low, n_feat)

    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, context_mask=None):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        hiddenvec = self.to_vec(down3)

        c = F.one_hot(c, num_classes=self.n_classes).type(torch.float)
        if context_mask is not None:
            c = c * (1.0 - context_mask.float().view(-1, 1))

        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        cond_h = torch.cat(
            [cemb1.squeeze(-1).squeeze(-1), temb1.squeeze(-1).squeeze(-1)], dim=1
        )
        cond_l = torch.cat(
            [cemb2.squeeze(-1).squeeze(-1), temb2.squeeze(-1).squeeze(-1)], dim=1
        )

        u0 = self.up0(hiddenvec)
        fused1 = cemb1 * u0 + temb1
        fused1 = self.film_after_fuse1(fused1, cond_h)
        u1 = self.up1(fused1, down3)
        fused2 = cemb2 * u1 + temb2
        fused2 = self.film_after_fuse2(fused2, cond_l)
        u2 = self.up2(fused2, down2)
        u3 = self.up3(u2, down1)
        return self.out(torch.cat((u3, x), 1))


ConditionalUnetMinAdaGN = ConditionalUnet_MinAdaGN


class EncoderUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(EncoderUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.out = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(2 * n_feat, n_feat), 
            nn.ReLU(),
            nn.Linear(n_feat, n_classes), 
        )

    def forward(self, x):
        # x is (noisy) image

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        out = self.out(hiddenvec)
        return out


class EncoderUnet_64(nn.Module):
    """64×64 噪声图像分类器（用于 classifier guidance），结构与最深编码一致。"""

    def __init__(self, in_channels: int = 3, n_feat: int = 256, n_classes: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, n_feat)
        self.down3 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.GELU())
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * n_feat, n_feat),
            nn.ReLU(),
            nn.Linear(n_feat, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        hiddenvec = self.to_vec(down3)
        return self.out(hiddenvec)


class UnconditionalUnet_64(nn.Module):
    """无类别条件的 64×64 ε 网络（仅时间步），供 classifier guidance 训练 unconditional DDPM。"""

    def __init__(self, in_channels: int = 3, n_feat: int = 256) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, n_feat)
        self.down3 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 8, 8),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x0 = self.init_conv(x)
        down1 = self.down1(x0)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        hiddenvec = self.to_vec(down3)

        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        u0 = self.up0(hiddenvec)
        u1 = self.up1(u0 + temb1, down3)
        u2 = self.up2(u1 + temb2, down2)
        u3 = self.up3(u2, down1)
        return self.out(torch.cat((u3, x0), 1))


class UnconditionalMinAdaGN(nn.Module):
    """
    无条件 ε 网络 + 仅依赖时间的 FiLM（结构对齐 Conditional MinAdaGN 解码，无类别嵌入）。
    用于 MinAdaGN + classifier guidance。
    """

    def __init__(self, in_channels: int = 3, n_feat: int = 256) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.cond_dim_high = 4 * n_feat
        self.cond_dim_low = 2 * n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )
        self.film_after_fuse1 = FiLM_AdaGN(self.cond_dim_high, 2 * n_feat)
        self.film_after_fuse2 = FiLM_AdaGN(self.cond_dim_low, n_feat)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x0 = self.init_conv(x)
        down1 = self.down1(x0)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        tv1 = temb1.squeeze(-1).squeeze(-1)
        tv2 = temb2.squeeze(-1).squeeze(-1)
        cond_h = torch.cat([tv1, tv1], dim=1)
        cond_l = torch.cat([tv2, tv2], dim=1)

        u0 = self.up0(hiddenvec)
        fused1 = u0 + temb1
        fused1 = self.film_after_fuse1(fused1, cond_h)
        u1 = self.up1(fused1, down2)
        fused2 = u1 + temb2
        fused2 = self.film_after_fuse2(fused2, cond_l)
        u2 = self.up2(fused2, down1)
        return self.out(torch.cat((u2, x0), 1))


class UnconditionalMinAdaGN_64(nn.Module):
    """64×64 无条件 MinAdaGN（仅时间 FiLM），供 guided + MinAdaGN。"""

    def __init__(self, in_channels: int = 3, n_feat: int = 256) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.cond_dim_high = 4 * n_feat
        self.cond_dim_low = 2 * n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, n_feat)
        self.down3 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 8, 8),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )
        self.film_after_fuse1 = FiLM_AdaGN(self.cond_dim_high, 2 * n_feat)
        self.film_after_fuse2 = FiLM_AdaGN(self.cond_dim_low, n_feat)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x0 = self.init_conv(x)
        down1 = self.down1(x0)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        hiddenvec = self.to_vec(down3)

        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        tv1 = temb1.squeeze(-1).squeeze(-1)
        tv2 = temb2.squeeze(-1).squeeze(-1)
        cond_h = torch.cat([tv1, tv1], dim=1)
        cond_l = torch.cat([tv2, tv2], dim=1)

        u0 = self.up0(hiddenvec)
        fused1 = u0 + temb1
        fused1 = self.film_after_fuse1(fused1, cond_h)
        u1 = self.up1(fused1, down3)
        fused2 = u1 + temb2
        fused2 = self.film_after_fuse2(fused2, cond_l)
        u2 = self.up2(fused2, down2)
        u3 = self.up3(u2, down1)
        return self.out(torch.cat((u3, x0), 1))