import numpy as np
import torch
from torch import nn
from encoder import PositionalEncoder
from typing import Tuple

class NeRF(nn.Module):
    def __init__(
            self,
            d_input: int = 3,
            input_encoding_length: int = 10,
            d_view: int = 3, # assumes unit vector view direction.
            view_encoding_length: int = 4,
            n_layers: int = 8,
            h_size: int = 256,
            skip: Tuple[int] = (4,)
    ):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = nn.functional.relu
        self.input_encoder = PositionalEncoder(d_input, input_encoding_length, log=False)
        self.view_encoder = PositionalEncoder(d_view, view_encoding_length, log=False)

        encoded_input_size = self.input_encoder.d_output
        encoded_view_size = self.view_encoder.d_output
        self.layers = nn.ModuleList(
            [nn.Linear(encoded_input_size, h_size)] +
            [nn.Linear(h_size + (encoded_input_size if i in skip else 0), h_size) for i in range(n_layers - 1)]
        )

        # Bottleneck layers
        self.sigma_out = nn.Linear(h_size, 1)
        self.rgb_filters = nn.Linear(h_size, h_size)
        self.branch = nn.Linear(h_size + encoded_view_size, int(h_size / 2))
        self.output = nn.Linear(int(h_size / 2), 3)

    def forward(self, x: torch.Tensor, viewdir: torch.Tensor):
        # Encode input
        x = self.input_encoder(x)
        print(x)
        viewdir = self.view_encoder(viewdir)
        print(viewdir)
        x_input = x
        for i, layer in enumerate(self.layers):
            print(i)
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # Bottleneck
        sigma = self.sigma_out(x)
        x = self.rgb_filters(x)
        x = torch.concat([x, viewdir], dim=-1)
        x = self.act(self.branch(x))
        x = self.output(x)

        return torch.concat([x, sigma], dim=-1)


nerf = NeRF()
pos = torch.rand((3))
ori = torch.rand((3))
print(nerf(pos,ori))
for param in nerf.parameters():
    print(type(param), param.size())
