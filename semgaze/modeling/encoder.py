#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import sys
import math
from typing import Dict, List, Optional, Tuple, Type, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from semgaze.utils.common import build_2d_sincos_posemb, pair

# ****************************************************** #
#                      GAZE ENCODER                      #
# ****************************************************** #
class GazeEncoder(nn.Module):
    def __init__(self, token_dim=768, feature_dim=512, gaze_vec_dim=2):
        super().__init__()

        self.feature_dim = feature_dim
        self.token_dim = token_dim
        self.gaze_vec_dim = gaze_vec_dim

        base = models.resnet18(weights=None)  # type: ignore
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        dummy_head = torch.empty((1, 3, 224, 224))
        dummy_head = self.backbone(dummy_head)
        embed_dim = dummy_head.size(1)

        self.gaze_proj = nn.Sequential(
            nn.Linear(embed_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim),
        )
        self.pos_proj = nn.Linear(4, token_dim)

        self.gaze_predictor = nn.Sequential(
            nn.Linear(embed_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, gaze_vec_dim),  # 2 or 3 = number of outputs (x, y, z) unit vector
            nn.Tanh(),
        )

    def forward(self, head, head_bbox):
        b, n, p = head_bbox.shape
        b, n, c, h, w = head.shape

        head_bbox_emb = self.pos_proj(head_bbox.view(-1, p))  # (b*n, token_dim)
        gaze_emb = self.backbone(head.view(-1, c, h, w)).flatten(1, -1)  # (b*n, embed_dim)

        gaze_token = self.gaze_proj(gaze_emb) + head_bbox_emb  # (b*n, token_dim)
        gaze_token = gaze_token.view(b, n, -1)  # (b, n, token_dim)

        gaze_vec = self.gaze_predictor(gaze_emb)  # (b*n, gaze_vec_dim)
        gaze_vec = F.normalize(gaze_vec, p=2, dim=1)  # normalize to unit vector
        gaze_vec = gaze_vec.view(b, n, -1)  # (b, n, gaze_vec_dim)

        return gaze_token, gaze_vec

# ****************************************************** #
#                      VIT ENCODER                       #
# ****************************************************** #
# The implementation of the ViT encoder is taken from the MultiMAE repository (https://github.com/EPFL-VILAB/MultiMAE).
class ViTEncoder(nn.Module):
    def __init__(
        self,
        num_global_tokens: int = 1,
        token_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        # Add global tokens
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, token_dim))
        nn.init.trunc_normal_(self.global_tokens, std=0.02)

        # Add encoder layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.encoder = nn.Sequential(
            *[
                TransformerBlock(dim=token_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, use_qkv_bias=use_qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[i])
                for i in range(depth)
            ]
        )

        # Initialize weights
        self.apply(self._init_weights)
        # Initialize the weights of Q, K, V separately
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and ("qkv" in name):
                val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                nn.init.uniform_(m.weight, -val, val)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def __len__(self):
        return len(self.encoder)

    def forward(self, input_tokens: torch.Tensor, return_all_layers: bool = True):
        # Add global tokens to input tokens
        global_tokens = einops.repeat(self.global_tokens, "() n d -> b n d", b=len(input_tokens))
        input_tokens = torch.cat([input_tokens, global_tokens], dim=1)

        # Pass tokens through Transformer
        if not return_all_layers:
            encoder_tokens = self.encoder(input_tokens)
        else:
            # Optionally access every intermediate layer
            encoder_tokens = []
            tokens = input_tokens
            for block in self.encoder:
                tokens = block(tokens)
                encoder_tokens.append(tokens)

        return encoder_tokens


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, use_qkv_bias=False, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, use_qkv_bias=use_qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rate=drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop_rate=drop_rate)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_rate=0.0):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, use_qkv_bias=False, attn_drop_rate=0.0, proj_drop_rate=0.0):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=use_qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        #self.attn = attn.detach() # to store attention matrix
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# ****************************************************** #
#                SPATIAL INPUT TOKENIZER                 #
# ****************************************************** #
class SpatialInputTokenizer(nn.Module):
    """Tokenizer for spatial inputs, like images or heatmaps.
    Creates tokens from patches over the input image.

    :param num_channels: Number of input channels of the image/feature map
    :param stride_level: Stride level compared to the full-sized image (e.g. 4 for 1/4th the size of the image).
    :param patch_size: Int or tuple of the patch size over the full image size. Patch size for smaller inputs will be computed accordingly.
    :param token_dim: Dimension of output tokens.
    :param use_sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings.
    :param is_learnable_pos_emb: Set to True to learn positional embeddings instead.
    :param image_size: Default image size. Used to initialize size of positional embeddings.
    """

    def __init__(
        self,
        num_channels: int,
        stride_level: int,
        patch_size: Union[int, Tuple[int, int]],
        token_dim: int = 768,
        use_sincos_pos_emb: bool = True,
        is_learnable_pos_emb: bool = False,
        image_size: Union[int, Tuple[int]] = 224,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size = pair(patch_size)
        self.token_dim = token_dim
        self.use_sincos_pos_emb = use_sincos_pos_emb
        self.is_learnable_pos_emb = is_learnable_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])

        self.P_H = max(1, self.patch_size[0] // stride_level)
        self.P_W = max(1, self.patch_size[1] // stride_level)

        self._init_pos_emb()
        self.proj = nn.Conv2d(in_channels=self.num_channels, out_channels=self.token_dim, kernel_size=(self.P_H, self.P_W), stride=(self.P_H, self.P_W))

    def _init_pos_emb(self):
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_pos_emb = self.image_size[0] // (self.stride_level * self.P_H)
        w_pos_emb = self.image_size[1] // (self.stride_level * self.P_W)

        if self.use_sincos_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=h_pos_emb, w=w_pos_emb, embed_dim=self.token_dim)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=self.is_learnable_pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.token_dim, h_pos_emb, w_pos_emb))
            nn.init.trunc_normal_(self.pos_emb, mean=0.0, std=0.02, a=-2.0, b=2.0)

    def forward(self, x):
        # input.shape = BxCxHxW >> output.shape = BxNxD (where N=n_tokens, D=token_dim)
        B, C, H, W = x.shape

        assert (H % self.P_H == 0) and (W % self.P_W == 0), f"Image size {H}x{W} must be divisible by patch size {self.P_H}x{self.P_W}"
        N_H, N_W = H // self.P_H, W // self.P_W  # Number of patches in height and width

        # Create tokens [B, C, PH, PW] >> [B, D, H, W] >> [B, (H*W), D]
        x_tokens = einops.rearrange(self.proj(x), "b d h w -> b (h w) d")

        # Create positional embedding
        x_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode="bicubic", align_corners=False)
        x_pos_emb = einops.rearrange(x_pos_emb, "b d h w -> b (h w) d")

        # Add patches and positional embeddings
        x = x_tokens + x_pos_emb

        return x


