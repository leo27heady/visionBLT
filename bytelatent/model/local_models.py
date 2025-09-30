# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn
import torch.nn as nn
from torchvision import transforms
from pydantic import ConfigDict
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask
from xformers.ops import AttentionBias

from bytelatent.base_transformer import (
    BaseTransformerArgs,
    VisionModelArgs,
    InitStdFactor,
    RotaryEmbedding,
    RotaryEmbeddingNd,
    TransformerBlock,
)
from bytelatent.model.latent_transformer import CrossAttention
from bytelatent.model.utils import create_causal_mask, downsample
from bytelatent.tokenizers.blt_tokenizer import BOE_ID

logger = logging.getLogger()
try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm

    RMSNorm = FusedRMSNorm
except (ImportError, ModuleNotFoundError):
    logging.debug("Apex not found. Using nn.RMSNorm")
    RMSNorm = nn.RMSNorm


class LocalModelArgs(BaseTransformerArgs):
    model_config = ConfigDict(extra="forbid")
    # Override defaults
    attn_impl: str | None = "xformers"
    attn_bias_type: str | None = "local_block_causal"
    
    vision: VisionModelArgs

    # Local encoder specific dimensions
    dropout: float
    vocab_size: int
    patch_size: float
    sliding_window: int | None
    use_rope: bool
    cross_attn_encoder: bool | None
    cross_attn_decoder: bool | None
    cross_attn_k: int | None
    cross_attn_init_by_pooling: bool
    patching_mode: str
    use_local_encoder_transformer: bool
    downsampling_by_pooling: str | None
    encoder_hash_byte_group_size: Any | None = None
    cross_attn_all_layers_encoder: bool = False
    cross_attn_all_layers_decoder: bool = False
    cross_attn_nheads: int | None

    dim_token_emb: int
    dim_patch_emb: int | None


class LocalModelBase(nn.Module):
    def __init__(self, args: LocalModelArgs):
        super().__init__()

        self.dim = args.dim
        self.dropout = args.dropout
        self.vocab_size = args.vocab_size
        self.patch_size = args.patch_size
        self.dim_patch_emb = args.dim_patch_emb

        self.attn_impl = args.attn_impl
        self.sliding_window = args.sliding_window
        self.use_rope = args.use_rope
        self.init_std_factor = args.init_std_factor
        self.cross_attn_encoder = getattr(args, "cross_attn_encoder", None)
        self.cross_attn_decoder = getattr(args, "cross_attn_decoder", None)
        self.cross_attn_k = getattr(args, "cross_attn_k", None)
        self.eos_id = args.eos_id

        self.boe_id = BOE_ID

        self.layers = nn.ModuleList(
            [TransformerBlock(args) for _ in range(args.n_layers)]
        )

        if not self.use_rope:
            self.pos_embeddings = nn.Embedding(args.max_length, args.dim)
        else:
            self.rope = RotaryEmbeddingNd(
                additional_dims=(args.vision.latent_height, args.vision.latent_width), 
                feature_dim=args.head_dim or args.dim // args.n_heads, 
                max_seqlen=args.max_seqlen,
                base=args.rope_theta,
                rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
            )
            self.pos_embeddings = None

        self.token_embedding_projection = (
            nn.Linear(args.dim_token_emb, args.dim, bias=False)
            if hasattr(args, "dim_token_emb") and args.dim_token_emb != self.dim
            else None
        )

        self.patch_embedding_projection = self._create_patch_projection(args)

    def _should_create_patch_projection(self, args: LocalModelArgs):
        dimension_mismatch = (
            getattr(args, "dim_patch_emb") and args.dim_patch_emb != self.dim
        )

        # Check cross attention conditions
        cross_attn_conditions = (
            args.cross_attn_encoder and args.cross_attn_init_by_pooling
        ) or (args.cross_attn_decoder and args.cross_attn_init_by_pooling)

        return dimension_mismatch or cross_attn_conditions

    def _create_patch_projection(self, args):
        if not self._should_create_patch_projection(args):
            return None

        output_dim = args.dim_token_emb * (self.cross_attn_k or 1)

        return nn.Linear(
            in_features=args.dim_patch_emb,
            out_features=output_dim,
            bias=False,
        )

    def init_weights(self, init_std=None):
        self.rope.reset_parameters()
        if hasattr(self, "norm"):
            self.norm.reset_parameters()

        init_std = init_std or (self.dim ** (-0.5))
        if self.pos_embeddings is not None:
            nn.init.trunc_normal_(
                self.pos_embeddings.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(None, factor)

        if hasattr(self, "image_encoder"):
            self.image_encoder.init_weights(init_std)

        if hasattr(self, "image_decoder"):
            self.image_decoder.init_weights(init_std)

        if hasattr(self, "head"):
            self.head.init_weights(init_std)

        if self.token_embedding_projection is not None:
            nn.init.trunc_normal_(
                self.token_embedding_projection.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        if self.patch_embedding_projection is not None:
            patch_emb_std = self.dim_patch_emb ** (-0.5)
            nn.init.trunc_normal_(
                self.patch_embedding_projection.weight,
                mean=0.0,
                std=patch_emb_std,
                a=-3 * patch_emb_std,
                b=3 * patch_emb_std,
            )

        if self.cross_attn_layers is not None:
            for depth, layer in enumerate(self.cross_attn_layers):
                factor = {
                    InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                    InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                    InitStdFactor.DIM_RATIO: self.dim / 4096,
                    InitStdFactor.DISABLED: 1.0,
                }[self.init_std_factor]

                layer.init_weights(None, factor)


class ConvolutionUnit(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int, 
        padding: int, 
        norm_channels: int,
        convolution_type: str = "down"
    ):

        super().__init__()

        self.convolution = (nn.Conv2d if convolution_type == "down" else nn.ConvTranspose2d)(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(padding, padding
        ))
        self.normalization = nn.GroupNorm(norm_channels, out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.convolution(imgs)
        x = self.normalization(x)
        x = self.activation(x)
        return x
    
    def init_weights(self, init_std):
        nn.init.trunc_normal_(
            self.convolution.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        self.normalization.reset_parameters()


class ImageEncoder(nn.Module):
    def __init__(self, args: LocalModelArgs):
        super().__init__()

        self.preprocessing = transforms.Normalize(mean=[0.5, ], std=[0.5, ])

        # Prepare channel dim
        self.prepare_dim = ConvolutionUnit(
            in_channels=args.vision.img_channels, out_channels=args.dim, 
            kernel_size=3, stride=1, padding=1, norm_channels=args.vision.norm_channels
        )
        
        # Process image
        self.process = ConvolutionUnit(
            in_channels=args.dim, out_channels=args.dim, 
            kernel_size=3, stride=1, padding=1, norm_channels=args.vision.norm_channels
        )

        # Downsample 2x
        self.downsample = ConvolutionUnit(
            in_channels=args.dim, out_channels=args.dim, 
            kernel_size=4, stride=2, padding=1, norm_channels=args.vision.norm_channels
        )
    
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.preprocessing(imgs)
        x = self.prepare_dim(x)
        x = self.process(x)
        x = self.downsample(x)
        return x
    
    def init_weights(self, init_std):
        self.prepare_dim.init_weights(init_std)
        self.process.init_weights(init_std)
        self.downsample.init_weights(init_std)


class LocalEncoder(LocalModelBase):
    def __init__(self, args: LocalModelArgs):
        super().__init__(args)

        self.apply_transformer = args.use_local_encoder_transformer
        self.downsampling_by_pooling = args.downsampling_by_pooling
        self.expects_hash_embeddings = args.encoder_hash_byte_group_size is not None
        self.cross_attn_encoder = args.cross_attn_encoder
        self.cross_attn_all_layers_encoder = args.cross_attn_all_layers_encoder
        self.cross_attn_init_by_pooling = args.cross_attn_init_by_pooling
        self.cross_attn_nheads = args.cross_attn_nheads

        # Image encoder
        self.image_encoder = ImageEncoder(args)

        # Cross-attention
        if self.cross_attn_encoder:
            self.cross_attn_layers = torch.nn.ModuleList()
            layers_to_add = args.n_layers if self.cross_attn_all_layers_encoder else 1
            for _ in range(layers_to_add):
                self.cross_attn_layers.append(
                    CrossAttention(
                        dim=self.dim,
                        head_dim=self.dim // self.cross_attn_nheads,
                        n_heads=self.cross_attn_nheads,
                        n_kv_heads=self.cross_attn_nheads,
                        norm_eps=args.norm_eps,
                    )
                )

    def forward(
        self,
        frames: torch.Tensor,
        mask: Union["BlockMask", "AttentionBias", torch.Tensor, str],
        cross_mask: torch.Tensor,
        patch_embeds: Optional[torch.Tensor] = None,
        num_patches: Optional[int] = None,
        patch_ids: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        """ """
        bs, seqlen, channels = frames.shape

        freqs_cis = self.rope(seqlen=seqlen) if self.use_rope else None
        h = F.dropout(frames, p=self.dropout, training=self.training)

        for i, layer in enumerate(self.layers):
            h = layer(h, mask=mask, freq_cis=freqs_cis, attn_impl=self.attn_impl)
            # check if cross attention should be applied to either all layer or only the last layer
            if self.cross_attn_encoder and (
                i == len(self.layers) - 1 or self.cross_attn_all_layers_encoder
            ):
                patch_embeds = self.apply_cross_attention(
                    h, patch_embeds, i, bs, num_patches, patch_ids, cross_mask
                )

        h_residual = patch_embeds if self.cross_attn_encoder else None
        return (h, h_residual), cache

    def apply_cross_attention(
        self, h, patch_embeds, layer_idx, bs, num_patches, patch_ids, cross_mask
    ):
        # apply pooling and project
        if self.cross_attn_init_by_pooling and patch_embeds is None:
            patch_embeds = downsample(
                h,
                num_patches,
                patch_ids=patch_ids,
                downsampling_by_pooling=self.downsampling_by_pooling,
                patch_size=self.patch_size,
            )
            if self.patch_embedding_projection is not None:
                patch_embeds = self.patch_embedding_projection(patch_embeds)
                patch_embeds = patch_embeds.reshape(
                    bs, patch_embeds.shape[1] * self.cross_attn_k, self.dim
                )

        layer_idx = layer_idx if self.cross_attn_all_layers_encoder else 0
        patch_embeds_cross = self.cross_attn_layers[layer_idx](
            x=patch_embeds,
            kv=h,
            mask=cross_mask,
        )
        return patch_embeds + patch_embeds_cross


class ImageDecoder(nn.Module):
    def __init__(self, args: LocalModelArgs):
        super().__init__()

        # Process image
        self.process = ConvolutionUnit(
            in_channels=args.dim, out_channels=args.dim, 
            kernel_size=3, stride=1, padding=1, norm_channels=args.vision.norm_channels
        )

        # Upsample 2x
        self.upsample = ConvolutionUnit(
            in_channels=args.dim, out_channels=args.dim, 
            kernel_size=4, stride=2, padding=1, norm_channels=args.vision.norm_channels,
            convolution_type="up"
        )
    
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.process(imgs)
        x = self.upsample(x)
        return x
    
    def init_weights(self, init_std):
        self.process.init_weights(init_std)
        self.upsample.init_weights(init_std)
    

class PredictionHead(nn.Module):
    def __init__(self, args: LocalModelArgs):
        super().__init__()

        self.dropout = args.dropout

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(
            args.dim,
            args.vision.pixel_vocab_size,  # 1
            bias=False,
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_preds = self.norm(h)
        h_preds = F.dropout(h_preds, p=self.dropout, training=self.training)
        h_preds = self.output(h_preds)
        h_preds = h_preds.float()
        # h_preds = h_preds.reshape(h_preds.shape[0], 10, 64, 64, 1).permute(0, 1, 4, 2, 3)
        return h_preds
    
    def init_weights(self, init_std):
        self.norm.reset_parameters()
        
        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )


class LocalDecoder(LocalModelBase):
    def __init__(self, args: LocalModelArgs):
        super().__init__(args)

        # Model configuration flags
        self.cross_attn_decoder = args.cross_attn_decoder
        self.cross_attn_all_layers_decoder = args.cross_attn_all_layers_decoder
        self.cross_attn_init_by_pooling = args.cross_attn_init_by_pooling
        self.cross_attn_nheads = args.cross_attn_nheads

        # Cross-attention
        if self.cross_attn_decoder:
            self.cross_attn_layers = torch.nn.ModuleList()
            layers_to_add = args.n_layers if self.cross_attn_all_layers_decoder else 1
            for _ in range(layers_to_add):
                self.cross_attn_layers.append(
                    CrossAttention(
                        dim=self.dim,
                        head_dim=self.dim // self.cross_attn_nheads,
                        n_heads=self.cross_attn_nheads,
                        n_kv_heads=self.cross_attn_nheads,
                        norm_eps=args.norm_eps,
                    )
                )

        # Image encoder
        self.image_decoder = ImageDecoder(args)

        # Head
        self.head = PredictionHead(args)

    def forward(
        self,
        embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
        mask: Union["BlockMask", "AttentionBias", torch.Tensor, str],
        cross_mask: torch.Tensor,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        bs, seqlen, channels = embeds.shape
        h = embeds

        if self.patch_embedding_projection is not None:
            patch_embeds = self.patch_embedding_projection(patch_embeds)
            if self.cross_attn_k is not None:
                patch_embeds = patch_embeds.reshape(
                    bs, patch_embeds.shape[1] * self.cross_attn_k, self.dim
                )

        if patch_embeds is not None and not self.cross_attn_decoder:
            h = h + patch_embeds

        freqs_cis = self.rope(seqlen=seqlen) if self.use_rope else None

        h = F.dropout(h, p=self.dropout, training=self.training)
        for i, layer in enumerate(self.layers):
            if self.cross_attn_decoder and (
                i == 0 or self.cross_attn_all_layers_decoder
            ):
                # Use cross attention to extract info from patch_embeds into h
                h_cross = self.cross_attn_layers[i](
                    x=h,
                    kv=patch_embeds,
                    mask=cross_mask,
                )
                h = h + h_cross

            h = layer(h, mask=mask, freq_cis=freqs_cis, attn_impl=self.attn_impl)

        return h, cache
