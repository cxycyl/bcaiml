import numpy as np
import timm
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import sys
import os
import torch.nn.init as init
sys.path.append('.')
from extractor.high_frequency_feature_extraction import HighDctFrequencyExtractor
from extractor.low_frequency_feature_extraction import LowDctFrequencyExtractor
import math
from functools import partial



class ConvNeXt(timm.models.convnext.ConvNeXt):
    def __init__(self, conv_pretrain=False):
        super(ConvNeXt, self).__init__(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
        if conv_pretrain:
            print("Load Convnext pretrain.")
            model = timm.create_model('convnext_tiny', pretrained=True)
            self.load_state_dict(model.state_dict())
        original_first_layer = self.stem[0]
        new_first_layer = nn.Conv2d(6, original_first_layer.out_channels,
                                    kernel_size=original_first_layer.kernel_size, stride=original_first_layer.stride,
                                    padding=original_first_layer.padding, bias=False)
        new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
        new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(new_first_layer.weight[:, 3:, :, :])
        self.stem[0] = new_first_layer

    def forward_features(self, x):
        x = self.stem(x)
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        x = self.norm_pre(x)
        return x, out

    def forward(self, image, mask=None, *args, **kwargs):

        feature, out = self.forward_features(image)

        return feature, out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.float()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class UpsampleConcatConvSegformer(nn.Module):
    def __init__(self):
        super(UpsampleConcatConvSegformer, self).__init__()
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(512, 320, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, inputs):
        # 上采样
        x1, x2, x3, x4 = inputs
        up2 = self.upsample1(x2)
        up3 = self.upsample2(x3)
        up4 = self.upsample3(x4)

        x = torch.cat([x1, up2, up3, up4], dim=1)
        return x


class MixVisionTransformer(nn.Module):
    def __init__(self, pretrain_path=None, img_size=512, patch_size=4, in_chans=3, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])

        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        if pretrain_path is not None:
            print("Load segformer pretrain pth.")
            self.load_state_dict(torch.load(pretrain_path),
                                 strict=False)
        original_first_layer = self.patch_embed1.proj
        new_first_layer = nn.Conv2d(6, original_first_layer.out_channels,
                                    kernel_size=original_first_layer.kernel_size, stride=original_first_layer.stride,
                                    padding=original_first_layer.padding, bias=False)
        new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]

        new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(new_first_layer.weight[:, 3:, :, :])
        self.patch_embed1.proj = new_first_layer

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return x, outs

    def forward(self, x):
        x, outs = self.forward_features(x)
        return x, outs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()

        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape

        if (input_h > self.output_h) or (input_w > self.output_w):
            self.stride_h = input_h // self.output_h
            self.stride_w = input_w // self.output_w
            self.kernel_size = (
            input_h - (self.output_h - 1) * self.stride_h, input_w - (self.output_w - 1) * self.stride_w)

            if self.pool_type == 'avg':
                y = nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
            else:
                y = nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
        else:
            y = F.interpolate(
                x,
                size=(self.output_h, self.output_w),
                mode='bilinear',
                align_corners=True
            )

        return y


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)


    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))




class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        super(CrossAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h
        self.eps = 1e-8

        # key, query, value projections for all heads
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection

        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]

        # Self-Attention
        rgb_fea_flat = self.LN1(rgb_fea_flat)
        q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1,
                                                                                                     3)  # (b_s, h, nq, d_k)
        k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3,
                                                                                                     1)  # (b_s, h, d_k, nk) K^T
        v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1,
                                                                                                     3)  # (b_s, h, nk, d_v)

        ir_fea_flat = self.LN2(ir_fea_flat)
        q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1,
                                                                                                  3)  # (b_s, h, nq, d_k)
        k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3,
                                                                                                  1)  # (b_s, h, d_k, nk) K^T
        v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1,
                                                                                                  3)  # (b_s, h, nk, d_v)

        att_vis = torch.matmul(q_ir, k_vis) / (np.sqrt(self.d_k) + self.eps)
        att_ir = torch.matmul(q_vis, k_ir) / (np.sqrt(self.d_k) + self.eps)

        # get attention matrix
        att_vis = torch.softmax(att_vis, -1)
        att_vis = self.attn_drop(att_vis)
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)

        # output
        out_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
                                                                                     self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_vis = self.resid_drop(self.out_proj_vis(out_vis))  # (b_s, nq, d_model)
        out_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
                                                                                  self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_ir = self.resid_drop(self.out_proj_ir(out_ir))  # (b_s, nq, d_model)

        return [out_vis, out_ir]


class CrossTransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        super(CrossTransformerBlock, self).__init__()
        self.loops = loops_num
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                     nn.GELU(),
                                     nn.Linear(block_exp * d_model, d_model),
                                     nn.Dropout(resid_pdrop),
                                     )
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    nn.GELU(),
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )
        self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                 nn.GELU(),
                                 nn.Linear(block_exp * d_model, d_model),
                                 nn.Dropout(resid_pdrop),
                                 )

        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        bs, nx, c = rgb_fea_flat.size()
        h = w = int(math.sqrt(nx))

        for loop in range(self.loops):
            rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            rgb_att_out = self.coefficient1(rgb_fea_flat) + self.coefficient2(rgb_fea_out)
            ir_att_out = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)
            rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN2(rgb_att_out)))
            ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))

        return [rgb_fea_flat, ir_fea_flat]

class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out


class NiNfusion(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(NiNfusion, self).__init__()

        self.concat = Concat(dimension=1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.concat(x)
        y = self.act(self.conv(y))

        return y

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class LearnableWeights(nn.Module):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = 1 - self.w1

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out


class TransformerFusionBlock(nn.Module):
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1,
                 attn_pdrop=0.1, resid_pdrop=0.1):
        super(TransformerFusionBlock, self).__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        d_k = d_model
        d_v = d_model

        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))


        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        self.apply(self._init_weights)

        self.crosstransformer = nn.Sequential(
            *[CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in
              range(n_layer)])

        # Concat
        self.concat = Concat(dimension=1)

        # conv1x1
        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis

        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir

        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])

        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
        else:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
        new_rgb_fea = rgb_fea_CFE + rgb_fea
        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        new_ir_fea = ir_fea_CFE + ir_fea

        new_fea = self.concat([new_rgb_fea, new_ir_fea])
        new_fea = self.conv1x1_out(new_fea)

        return new_fea


class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x.permute(0, 2, 1).view(B, self.embed_dim, H, W)


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MLPDecoder(nn.Module):
    def __init__(self, in_channels, embedding_dim, num_classes=1):
        super().__init__()
        self.linear_c1 = MLP(in_channels[0], embedding_dim)
        self.linear_c2 = MLP(in_channels[1], embedding_dim)
        self.linear_c3 = MLP(in_channels[2], embedding_dim)
        self.linear_c4 = MLP(in_channels[3], embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1
        )
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        B, _, H1, W1 = c1.shape

        c2 = F.interpolate(c2, size=(H1, W1), mode='bilinear', align_corners=False)
        c3 = F.interpolate(c3, size=(H1, W1), mode='bilinear', align_corners=False)
        c4 = F.interpolate(c4, size=(H1, W1), mode='bilinear', align_corners=False)

        c1_emb = self.linear_c1(c1)
        c2_emb = self.linear_c2(c2)
        c3_emb = self.linear_c3(c3)
        c4_emb = self.linear_c4(c4)

        fused = torch.cat([c1_emb, c2_emb, c3_emb, c4_emb], dim=1)
        fused = self.linear_fuse(fused)
        fused = self.dropout(fused)
        pred = self.linear_pred(fused)
        return pred

class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, target):
        B, C, H, W = inputs.shape
        inputs = inputs.view(B, -1)
        target = target.view(B, -1).float()

        bce_loss = F.binary_cross_entropy_with_logits(inputs, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        return focal_loss.mean()

class Dice_Loss(nn.Module):
    def __init__(self, beta=1, smooth=1e-5):
        super().__init__()
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, target):
        inputs = inputs.flatten(1)  # (B, H*W)
        target = target.flatten(1).float()  # (B, H*W)

        intersection = (inputs * target).sum(1)
        denominator = inputs.sum(1) + target.sum(1)

        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()

@MODELS.register_module()
class BCAIML(nn.Module):
    def __init__(self, seg_pretrain_path=None, conv_pretrain=False, image_size=512):
        super(BCAIML, self).__init__()
        self.convnext = ConvNeXt(conv_pretrain)
        self.segformer = MixVisionTransformer(seg_pretrain_path)
        self.low_dct = LowDctFrequencyExtractor()
        self.high_dct = HighDctFrequencyExtractor()
        self.resize = nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=True)
        self.channel_align_convnext = nn.ModuleList([
            nn.Conv2d(96, 256, kernel_size=1),
            nn.Conv2d(192, 256, kernel_size=1),
            nn.Conv2d(384, 256, kernel_size=1),
            nn.Conv2d(768, 256, kernel_size=1)
        ])
        self.channel_align_segformer = nn.ModuleList([
            nn.Conv2d(64, 256, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.Conv2d(320, 256, kernel_size=1),
            nn.Conv2d(512, 256, kernel_size=1)
        ])

        self.bcaf_blocks = nn.ModuleList([
            TransformerFusionBlock(
                d_model=256,
                vert_anchors=16,
                horz_anchors=16,
                h=8,
                block_exp=4,
                n_layer=1
            )
            for i in range(4)
        ])

        self.mlp_decoder = MLPDecoder(
            in_channels=[256, 256, 256, 256],
            embedding_dim=256,
            num_classes=1
        )

        self.focal_loss = Focal_Loss(alpha=0.5, gamma=2)
        self.dice_loss = Dice_Loss(beta=1, smooth=1e-5)
        self.loss_weights = {
            "focal": 0.5,
            "dice": 0.5
        }


    def forward(self, image, mask, *args, **kwargs):
        high_freq = self.high_dct(image)
        low_freq = self.low_dct(image)
        input_high = torch.concat([image, high_freq], dim=1)
        input_low = torch.concat([image, low_freq], dim=1)
        _, outs1 = self.convnext(input_high)
        _, outs2 = self.segformer(input_low)

        aligned_outs1 = []
        aligned_outs2 = []
        for i in range(4):
            aligned_outs1.append(self.channel_align_convnext[i](outs1[i]))
            aligned_outs2.append(self.channel_align_segformer[i](outs2[i]))

        fused_features = []
        for i in range(4):
            con_fea = aligned_outs1[i]
            seg_fea = aligned_outs2[i]
            fused_fea = self.bcaf_blocks[i]([con_fea, seg_fea])
            fused_features.append(fused_fea)

        pred_mask = self.mlp_decoder(fused_features)
        pred_mask = self.resize(pred_mask)

        focal_loss = self.focal_loss(pred_mask, mask)
        dice_loss = self.dice_loss(torch.sigmoid(pred_mask), mask)

        final_loss = (
                self.loss_weights["focal"] * focal_loss +
                self.loss_weights["dice"] * dice_loss
        )


        loss = final_loss
        pred_mask = pred_mask.float()
        mask_pred = torch.sigmoid(pred_mask)

        output_dict = {
            # loss for backward
            "backward_loss": loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": mask_pred,
            # predicted binaray label, will calculate for metrics automatically
            "pred_label": None,

            # ----values below is for visualization----
            # automatically visualize with the key-value pairs
            "visual_loss": {
                "predict_loss": loss,
            },

            "visual_image": {
                "pred_mask": mask_pred,
            }
            # -----------------------------------------
        }
        return output_dict


if __name__ == "__main__":
    print(MODELS)

# 测试用例
# if __name__ == "__main__":
#
#     # 模拟输入数据
#     batch_size = 2
#     image_channels = 3
#     image_size = 512
#     image = torch.randn(batch_size, image_channels, image_size, image_size)
#     mask = torch.randint(0, 2, (batch_size, 1, image_size, image_size)).float()
#
#     # 初始化模型
#     model = Mesorch(seg_pretrain_path=None, conv_pretrain=False, image_size=image_size)
#
#     try:
#         # 前向传播测试
#         output = model(image, mask)
#         print("前向传播测试成功！")
#         print("输出结果字典的键：", output.keys())
#         print("预测掩码的形状：", output["pred_mask"].shape)
#         print("损失值：", output["backward_loss"])
#     except Exception as e:
#         print("前向传播测试失败！错误信息如下：")
#         print(e)