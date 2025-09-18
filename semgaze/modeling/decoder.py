#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import math
from typing import Tuple, Type

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ****************************************************** #
#                      GAZE DECODER                      #
# ****************************************************** #
# The implementation of the Gaze Decoder is adapted from the Segment Anything repository (https://github.com/facebookresearch/segment-anything).
class GazeDecoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        depth: int,
        num_heads: int,
        label_emb_dim: int,
    ) -> None:
        """
        Predicts gaze heatmaps and label embeddings given a image and 
        gaze tokens, using a transformer architecture.

        Arguments:
          token_dim (int): the token dimension of the transformer
          depth (int): the number of layers in the transformer
          num_heads (int): the number of attention heads in the transformer
          label_emb_dim (int): the dimension of the output gaze label embedding
        """
        super().__init__()
        
        self.token_dim = token_dim
        self.depth = depth
        self.num_heads = num_heads
        self.label_emb_dim = label_emb_dim

        # Initialize a learnable `not a person` token
        self.not_a_person_embed = nn.Embedding(1, token_dim)
        
        # Two-way transformer for image-gaze cross-attention
        self.transformer = TwoWayTransformer(depth=depth, 
                                             token_dim=token_dim, 
                                             mlp_dim=2048, 
                                             num_heads=num_heads)

        # Upscaler for image features
        self.upscaler = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(token_dim, token_dim // 4, kernel_size=3, padding=1),
            LayerNorm2d(token_dim // 4),
            nn.GELU(),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(token_dim // 4, token_dim // 8, kernel_size=3, padding=1),
            nn.GELU(),
        )
        
        # MLPs for heatmap and label embeddings
        self.heatmap_mlp = MLP(token_dim, token_dim, token_dim // 8, 3)
        self.label_mlp = MLP(token_dim, token_dim, label_emb_dim, 6)
        
        
    def forward(
        self,
        image_tokens: torch.Tensor,
        gaze_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict gaze heatmaps and gaze label embeddings given image and gaze tokens.

        Arguments:
          image_tokens (torch.Tensor): the tokens encoding the scene image
          gaze_tokens (torch.Tensor): the tokens encoding people's head/gaze information

        Returns:
          torch.Tensor: batched predicted gaze heatmaps
          torch.Tensor: batched predicted gaze label embeddings
        """

        b, pn, pc = gaze_tokens.shape  # (b, n, c)
        b, ic, ih, iw = image_tokens.shape  # (b, c, h, w)
        
        # Add not-a-person token to gaze tokens
        not_person_token = self.not_a_person_embed.weight.unsqueeze(0).expand(b, -1, -1)  # (b, 1, c)
        gaze_tokens = torch.cat((gaze_tokens, not_person_token), dim=1)  # (b, n+1, c)

        # Run the transformer
        image_tokens = image_tokens.view(b, ic, ih*iw).permute(0, 2, 1) # (b, c, h, w) >> (b, h*w, c)
        gaze_tokens, image_tokens = self.transformer(gaze_tokens, image_tokens)  # (b, n, c) & (b, h*w, c)
        image_tokens = image_tokens.permute(0, 2, 1).view(b, ic, ih, iw) # (b, h*w, c) >> (b, c, h, w)
        
        # Upscale mask embeddings
        upscaled_image_tokens = self.upscaler(image_tokens) # (b, c', hm_h, hm_w)
        b, c, h, w = upscaled_image_tokens.shape

        # Filter out the not-a-person token
        gaze_tokens = gaze_tokens[:, :-1, :]  # (b, n, c)
        gaze_tokens = gaze_tokens.reshape(-1, pc) # (b*n, c)
        
        # Predict gaze heatmap
        gaze_heatmap_emb = self.heatmap_mlp(gaze_tokens) # (b*n, c')
        gaze_heatmap_emb = gaze_heatmap_emb.view(b, pn, -1) # (b, n, c')
        gaze_heatmap = (gaze_heatmap_emb @ upscaled_image_tokens.view(b, c, h * w)) # (b, n, hm_h*hm_w)
        gaze_heatmap = gaze_heatmap.view(b, -1, h, w) # (b, n, hm_h, hm_w)
        
        # Predict gaze label
        gaze_label_emb = self.label_mlp(gaze_tokens) # (b*n, 512)
        gaze_label_emb = F.normalize(gaze_label_emb, p=2, dim=1).view(b, pn, -1) # (b, n, 512)

        return gaze_heatmap, gaze_label_emb


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        token_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          token_dim (int): the dimension of the tokens
          num_heads (int): the number of heads for multihead attention. Must
            divide token_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        
        self.depth = depth
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    token_dim=token_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                )
            )

        self.final_attn_gaze_to_image = Attention(token_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(token_dim)

    def forward(
        self,
        queries: Tensor,
        context: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x token_dim x h x w for any h and w.
          gaze_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x token_dim for any N_points.

        Returns:
          torch.Tensor: the processed gaze_embedding
          torch.Tensor: the processed image_embedding
        """

        # Apply transformer blocks
        for layer in self.layers:
            queries, context = layer(queries=queries, context=context)

        # Apply the final attention layer from gaze tokens to image tokens
        attn_out = self.final_attn_gaze_to_image(q=queries, k=context, v=context)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, context


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer block with three layers: (1) cross attention of gaze to image 
        tokens, (2) mlp block on gaze tokens, and (3) cross attention of image to 
        gaze tokens.

        Arguments:
          token_dim (int): the dimension of the tokens
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
        """
        super().__init__()

        self.cross_attn_gaze_to_image = Attention(token_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(token_dim)

        self.mlp = MLPBlock(token_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(token_dim)

        self.norm4 = nn.LayerNorm(token_dim)
        self.cross_attn_image_to_gaze = Attention(token_dim, num_heads, downsample_rate=attention_downsample_rate)

    def forward(self, queries: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:

        # Cross attention block, tokens attending to image embedding
        attn_out = self.cross_attn_gaze_to_image(q=queries, k=context, v=context)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        attn_out = self.cross_attn_image_to_gaze(q=context, k=queries, v=queries)
        context = context + attn_out
        context = self.norm4(context)

        return queries, context


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.internal_dim = token_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide token_dim."

        self.q_proj = nn.Linear(token_dim, self.internal_dim)
        self.k_proj = nn.Linear(token_dim, self.internal_dim)
        self.v_proj = nn.Linear(token_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, token_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head) # inject fov here
        attn = torch.softmax(attn, dim=-1)
        
        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
    
    
class MLPBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
    
class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interpolate = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x

    def __repr__(self):
        return f"Interpolate(scale_factor={self.scale_factor}, mode={self.mode}, align_corners={self.align_corners})"