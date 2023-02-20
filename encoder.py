#!/usr/bin/env python3
import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, d_input: int, n_freqs: int, log: bool=False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log = log
        self.embed_fns = [lambda x: x]
        self.d_output = d_input * (1 + n_freqs * 2)

        if log:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        for f in freq_bands:
            self.embed_fns.append(lambda x, f=f: torch.sin(x * f))
            self.embed_fns.append(lambda x, f=f: torch.cos(x * f))

    def forward(self, x):
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)
