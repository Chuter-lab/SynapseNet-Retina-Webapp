"""UNet2d model from torch_em, vendored to avoid heavy dependency chain.

Source: https://github.com/constantinpape/torch-em (MIT License)
Only the 2D UNet and required base classes are included.
"""
from typing import Optional, Union

import torch
import torch.nn as nn


class UNetBase(nn.Module):
    def __init__(self, encoder, base, decoder, out_conv=None,
                 final_activation=None, postprocessing=None, check_shape=True):
        super().__init__()
        self.encoder = encoder
        self.base = base
        self.decoder = decoder
        self.return_decoder_outputs = False
        if out_conv is None:
            self._out_channels = self.decoder.out_channels
        else:
            self._out_channels = out_conv.out_channels
        self.out_conv = out_conv
        self.check_shape = check_shape
        self.final_activation = self._get_activation(final_activation)
        self.postprocessing = postprocessing

    @property
    def in_channels(self):
        return self.encoder.in_channels

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def depth(self):
        return len(self.encoder)

    def _get_activation(self, activation):
        if activation is None:
            return None
        if isinstance(activation, nn.Module):
            return activation
        if isinstance(activation, str):
            act = getattr(nn, activation, None)
            if act is None:
                raise ValueError(f"Invalid activation: {activation}")
            return act()
        raise ValueError(f"Invalid activation: {activation}")

    def _check_shape(self, x):
        spatial_shape = tuple(x.shape)[2:]
        depth = len(self.encoder)
        factor = [2**depth] * len(spatial_shape)
        if any(sh % fac != 0 for sh, fac in zip(spatial_shape, factor)):
            raise ValueError(f"Invalid shape for U-Net: {spatial_shape} not divisible by {factor}")

    def forward(self, x):
        if getattr(self, "check_shape", True):
            self._check_shape(x)

        self.encoder.return_outputs = True
        self.decoder.return_outputs = False

        x, encoder_out = self.encoder(x)
        x = self.base(x)
        x = self.decoder(x, encoder_inputs=encoder_out[::-1])

        if self.out_conv is not None:
            x = self.out_conv(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        if self.postprocessing is not None:
            x = self.postprocessing(x)
        return x


def get_norm_layer(norm, dim, channels, n_groups=32):
    if norm is None:
        return None
    if norm == "InstanceNorm":
        return nn.InstanceNorm2d(channels) if dim == 2 else nn.InstanceNorm3d(channels)
    elif norm == "BatchNorm":
        return nn.BatchNorm2d(channels) if dim == 2 else nn.BatchNorm3d(channels)
    elif norm == "GroupNorm":
        return nn.GroupNorm(min(n_groups, channels), channels)
    raise ValueError(f"Invalid norm: {norm}")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim, kernel_size=3, padding=1, norm="InstanceNorm"):
        super().__init__()
        conv = nn.Conv2d if dim == 2 else nn.Conv3d
        if norm is None:
            self.block = nn.Sequential(
                conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                get_norm_layer(norm, dim, in_channels),
                conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                get_norm_layer(norm, dim, out_channels),
                conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.block(x)


class ConvBlock2d(ConvBlock):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, dim=2, **kwargs)


class Encoder(nn.Module):
    def __init__(self, features, scale_factors, conv_block_impl, pooler_impl, **conv_block_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [conv_block_impl(inc, outc, **conv_block_kwargs)
             for inc, outc in zip(features[:-1], features[1:])]
        )
        self.poolers = nn.ModuleList([pooler_impl(f) for f in scale_factors])
        self.return_outputs = True
        self.in_channels = features[0]
        self.out_channels = features[-1]

    def __len__(self):
        return len(self.blocks)

    def forward(self, x):
        encoder_out = []
        for block, pooler in zip(self.blocks, self.poolers):
            x = block(x)
            encoder_out.append(x)
            x = pooler(x)
        if self.return_outputs:
            return x, encoder_out
        return x


class Upsampler2d(nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, mode="bilinear"):
        super().__init__()
        self.mode = mode
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, features, scale_factors, conv_block_impl, sampler_impl, **conv_block_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [conv_block_impl(inc, outc, **conv_block_kwargs)
             for inc, outc in zip(features[:-1], features[1:])]
        )
        self.samplers = nn.ModuleList(
            [sampler_impl(f, inc, outc) for f, inc, outc
             in zip(scale_factors, features[:-1], features[1:])]
        )
        self.return_outputs = False
        self.in_channels = features[0]
        self.out_channels = features[-1]

    def __len__(self):
        return len(self.blocks)

    def _crop(self, x, shape):
        shape_diff = [(xsh - sh) // 2 for xsh, sh in zip(x.shape, shape)]
        crop = tuple([slice(sd, xsh - sd) for sd, xsh in zip(shape_diff, x.shape)])
        return x[crop]

    def forward(self, x, encoder_inputs):
        for block, sampler, from_encoder in zip(self.blocks, self.samplers, encoder_inputs):
            x = sampler(x)
            x = block(torch.cat([x, self._crop(from_encoder, x.shape)], dim=1))
        return x


class UNet2d(UNetBase):
    """2D U-Net for segmentation."""
    def __init__(self, in_channels=1, out_channels=1, depth=4, initial_features=32,
                 gain=2, final_activation=None, check_shape=True, **conv_block_kwargs):
        features_encoder = [in_channels] + [initial_features * gain ** i for i in range(depth)]
        features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]
        scale_factors = depth * [2]

        out_conv = None if out_channels is None else nn.Conv2d(features_decoder[-1], out_channels, 1)

        super().__init__(
            encoder=Encoder(features_encoder, scale_factors, ConvBlock2d, nn.MaxPool2d, **conv_block_kwargs),
            decoder=Decoder(features_decoder, scale_factors[::-1], ConvBlock2d, Upsampler2d, **conv_block_kwargs),
            base=ConvBlock2d(features_encoder[-1], features_encoder[-1] * gain, **conv_block_kwargs),
            out_conv=out_conv,
            final_activation=final_activation,
            check_shape=check_shape,
        )
        self.init_kwargs = {
            "in_channels": in_channels, "out_channels": out_channels, "depth": depth,
            "initial_features": initial_features, "gain": gain, "final_activation": final_activation,
            **conv_block_kwargs,
        }
