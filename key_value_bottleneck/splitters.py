from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops as eo


class Splitter(nn.Module):
    pass


class RandomDownProjection(Splitter):
    def __init__(
        self,
        num_codebooks: int,
        num_channels: int,
        dim_keys: int,
    ):
        super().__init__()
        proj = F.normalize(
            torch.randn(num_codebooks, num_channels, dim_keys), dim=1, p=2
        )
        self.register_buffer("rand_proj", proj)

    def forward(self, x):
        x = torch.einsum("b...d,xdu->b...ux", x, self.rand_proj)
        x = eo.rearrange(x, "b ... u x -> b x ... u")
        return x


class LearnedDownProjection(Splitter):
    def __init__(
            self,
            num_codebooks: int,
            num_channels: int,
            dim_keys: int,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.num_channels = num_channels
        self.dim_keys = dim_keys
        self.proj = nn.Linear(in_features=num_channels, out_features=num_codebooks * dim_keys, bias=False)

    def forward(self, x):
        x = eo.rearrange(x, "b ... d1 -> b (...) d1")
        x = self.proj(x)
        x = eo.rearrange(x, "b n (dim_keys num_books) -> b num_books n dim_keys",
                         dim_keys=self.dim_keys,
                         num_books=self.num_codebooks)
        return x


class Chunker(Splitter):
    def __init__(
        self,
        num_codebooks: int,
        num_channels: int,
        dim_keys: Optional[int] = None,
    ):
        super(Chunker, self).__init__()
        assert (
            num_channels % num_codebooks == 0
        ), "Number of codebooks must be divisible by the number of channels."
        if dim_keys is None:
            dim_keys = num_channels // num_codebooks
        else:
            assert num_channels // num_codebooks == dim_keys
        self.num_codebooks = num_codebooks
        self.num_channels = num_channels
        self.dim_keys = dim_keys

    def forward(self, x: torch.Tensor):
        x = eo.rearrange(x, "b ... (c k) -> b c ... k", c=self.num_codebooks, k=self.dim_keys)
        return x
