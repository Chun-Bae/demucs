# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Conveniance wrapper to perform STFT and iSTFT"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math


def spectro(x, n_fft=512, hop_length=None, pad=0):
    *other, length = x.shape
    x = x.reshape(-1, length)
    z = th.stft(
        x,
        n_fft * (1 + pad),
        hop_length or n_fft // 4,
        window=th.hann_window(n_fft).to(x),
        win_length=n_fft,
        normalized=True,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


def ispectro(z, hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    x = th.istft(
        z,
        n_fft,
        hop_length,
        window=th.hann_window(win_length).to(z.real),
        win_length=win_length,
        normalized=True,
        length=length,
        center=True,
    )
    _, freqs, frame = z.shape
    return x.view(*other, length)


class ConvSTFT(nn.Module):
    def __init__(self, n_fft=4096, hop_length=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        n = th.arange(n_fft, dtype=th.float32)
        k = th.arange(n_fft // 2 + 1, dtype=th.float32)
        fourier_basis = 2 * math.pi * k[:, None] * n[None, :] / n_fft
        w = th.hann_window(n_fft)

        real_basis = th.cos(-fourier_basis) * w
        imag_basis = th.sin(-fourier_basis) * w
        basis = th.cat([real_basis, imag_basis], dim=0).unsqueeze(1) / math.sqrt(n_fft)

        self.register_buffer("basis", basis)

    def forward(self, x):
        pad_len = self.n_fft // 2
        x = F.pad(x, (pad_len, pad_len), mode="reflect")
        y = F.conv1d(x.unsqueeze(1), self.basis, stride=self.hop_length)
        return y


class ConvISTFT(nn.Module):
    def __init__(self, n_fft=4096, hop_length=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        n = th.arange(n_fft, dtype=th.float32)
        k = th.arange(n_fft // 2 + 1, dtype=th.float32)
        fourier_basis = 2 * math.pi * k[:, None] * n[None, :] / n_fft
        w = th.hann_window(n_fft)

        inv_real_basis = th.cos(fourier_basis) * 2
        inv_real_basis[0] /= 2
        inv_real_basis[-1] /= 2
        inv_imag_basis = -th.sin(fourier_basis) * 2

        inv_basis = th.cat([inv_real_basis, inv_imag_basis], dim=0).unsqueeze(1)
        inv_basis = inv_basis * w * math.sqrt(n_fft) / n_fft

        self.register_buffer("basis", inv_basis)
        self.register_buffer("w_sq", (w**2).view(1, 1, -1))

    def forward(self, y, length):
        batch, channels, frames = y.shape
        x_ola = F.conv_transpose1d(y, self.basis, stride=self.hop_length)
        ones = th.ones(1, 1, frames, device=y.device, dtype=y.dtype)
        w_sq_ola = F.conv_transpose1d(ones, self.w_sq, stride=self.hop_length)

        x_rec = x_ola / (w_sq_ola + 1e-8)
        pad_len = self.n_fft // 2

        x_rec = x_rec.squeeze(1)[:, pad_len : pad_len + length]
        return x_rec
