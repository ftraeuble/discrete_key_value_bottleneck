import string
from typing import Iterable, Optional, Callable, Tuple, List

import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from einops import rearrange, repeat
import tqdm

from key_value_bottleneck.utils import (
    gumbel_sample,
    laplace_smoothing,
    ema_inplace,
    sample_vectors,
    kmeans,
)
from key_value_bottleneck.splitters import Chunker, RandomDownProjection, \
    LearnedDownProjection


class CLIPModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        embedding = self.model.encode_image(x)
        embedding = embedding.float()
        return embedding


class SwavModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        with torch.cuda.amp.autocast():
            embedding = self.model(x)
        embedding = embedding.float()
        return embedding

# Legend:
#   b: batch size
#   c: number of codebooks
#   d: number of encoder output channels
#   k: dimension of keys
#   v: dimension of values
#   h, w: height or width
#   n: number of tokens
#   i: number of queries
#   j: number of key-value pairs


class KeyValueBottleneck(nn.Module):
    def __init__(
            self,
            num_codebooks: int,
            key_value_pairs_per_codebook: int,
            dim_keys: int,
            dim_values: int,
            init_mode: str = "random",
            kmeans_iters: int = 10,
            decay: float = 0.95,
            eps: float = 1e-5,
            threshold_ema_dead_code: float = 0.0,
            sample_codebook_temperature: float = 0.0,
            return_values_only: bool = True,
            topk: int = 1,
    ):
        super().__init__()
        assert init_mode in ["random", "kmeans"]

        self.num_codebooks = num_codebooks
        self.key_value_pairs_per_codebook = key_value_pairs_per_codebook
        self.dim_keys = dim_keys
        self.dim_values = dim_values
        self.decay = decay
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temperature = sample_codebook_temperature
        self.return_values_only = return_values_only
        self.topk = topk
        self.init_mode = init_mode
        self.kmeans_iters = kmeans_iters

        init_fn = torch.randn if self.init_mode == "random" else torch.zeros
        keys = init_fn(num_codebooks, key_value_pairs_per_codebook, dim_keys)
        values = torch.randn(num_codebooks, key_value_pairs_per_codebook, dim_values)

        self.count_cluster_size = True

        self.register_buffer("initted", torch.Tensor([self.init_mode != "kmeans"]))
        self.register_buffer(
            "cluster_size", torch.zeros(num_codebooks, key_value_pairs_per_codebook)
        )
        self.register_buffer(
            "cluster_size_counter",
            torch.zeros(num_codebooks, key_value_pairs_per_codebook),
        )
        self.register_buffer("keys_avg", keys.clone())
        self.register_buffer("keys", keys)

        self.values = nn.Parameter(values)
        self.update_keys = True

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        keys, cluster_size = kmeans(
            data, self.key_value_pairs_per_codebook, self.kmeans_iters
        )
        self.keys.data.copy_(keys)
        self.keys.data.copy_(keys.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask):
        # samples.shape = cik
        # mask.shape = cj
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.key_value_pairs_per_codebook),
            self.keys,
        )
        self.keys.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return
        # expired_codes.shape = cj
        # expired_codes[c, j] = True means that the pair at index j for codebook c must be killed.
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        # batch_samples.shape = cik
        # expired_codes.shape = cj
        self.replace(batch_samples, mask=expired_codes)

    def reset_cluster_size_counter(self):
        self.cluster_size_counter.data.fill_(0)

    def deactivate_counts(self):
        self.count_cluster_size = False

    def enable_update_keys(self):
        self.update_keys = True

    def disable_update_keys(self):
        self.update_keys = False

    def activate_counts(self):
        self.count_cluster_size = True

    @autocast(enabled=False)
    def forward(self, x):
        x = x.float()

        # x.shape = b c n k
        shape, dtype, device = x.shape, x.dtype, x.device
        assert x.ndim == 4
        b, c, n, k = shape

        flatten = rearrange(x, "b c n k -> c (b n) k")

        self.init_embed_(flatten)

        # dist.shape = cij, where
        #   c is the number of codebooks
        #   i is the number of search queries
        #   j is the number of key-value pairs.
        dist: torch.Tensor = -(  # noqa
                torch.einsum("cid,cid->ci", flatten, flatten)[..., None]
                - 2 * torch.einsum("cid,cjd->cij", flatten, self.keys)
                + torch.einsum("cjd,cjd->cj", self.keys, self.keys)[:, None, :]
        )

        if (
                not self.count_cluster_size
        ):  # If we deactivated to count selections the model shall only be able to select keys with count > 0
            # and disregard (key,value) pairs that haven't been used so far
            not_used_pairs_masked = self.cluster_size_counter == 0  # shape cj
            not_used_pairs_masked = not_used_pairs_masked[:, None, :].expand(
                -1, dist.shape[1], -1
            )
            # all zero-count pairs will get maximum distance and are thus not fetched (unless finite temperature)
            max_dist = torch.min(dist)  # dist is negative distance
            dist[
                not_used_pairs_masked == 1
                ] = max_dist

        # keys_ind.shape = c(i topk), where
        #   c is the number of codebooks
        #   i topk is the number of search queries times the number of topk keys to fetch.
        if not self.update_keys:
            forward_topk = self.topk
        else:
            forward_topk = 1
        # for ema updating keys only the closest should be used.
        keys_ind = gumbel_sample(
            dist, dim=-1, temperature=self.sample_codebook_temperature, topk=forward_topk
        )

        # dist_to_fetched_keys.shape = c(i topk), where
        #   c is the number of codebooks
        #   i topk is the number of search queries times the number of topk keys to fetch.
        dist = einops.repeat(dist, "c i j -> c i topk j", topk=forward_topk)
        dist_to_fetched_keys = torch.gather(dist, dim=3, index=keys_ind[..., None])[
            ..., 0
        ]

        counts = einops.repeat(self.cluster_size_counter, "c j -> c i topk j", i=b*n, topk=forward_topk)
        counts = torch.gather(counts, dim=3, index=keys_ind[..., None])[
            ..., 0
        ]

        keys_ind = einops.rearrange(keys_ind, "c i topk-> c (i topk)")

        # gather_ind_keys.shape = c(i topk)k
        gather_ind_keys = repeat(keys_ind, "c i -> c i k", k=self.dim_keys)
        # self.keys.shape = cjk
        # quantized_keys.shape = c(i topk)k
        quantized_keys = torch.gather(self.keys, dim=1, index=gather_ind_keys)
        # gather_ind_values.shape = c(i topk)v
        gather_ind_values = repeat(keys_ind, "c i -> c i v", v=self.dim_values)
        # self.values.shape = cjv
        quantized_values = torch.gather(self.values, dim=1, index=gather_ind_values)

        if self.update_keys:

            # keys_onehot.shape = c(i topk)j
            keys_onehot = F.one_hot(keys_ind, self.key_value_pairs_per_codebook).type(dtype)

            # cluster_size.shape = cj
            # cluster_size[c, j] is the number of queries that attached to the key at index j
            # for codebook at index c.
            cluster_size = keys_onehot.sum(1)
            ema_inplace(self.cluster_size, cluster_size, self.decay)

            # keys_sum[c, :, j] is the average representation (for the batch)
            # generated by the encoder for the key at index j and codebook at index c.
            flatten = einops.repeat(flatten, "c n k -> c (n topk) k", topk=forward_topk)
            keys_sum = torch.einsum("cik,cij->ckj", flatten, keys_onehot)
            ema_inplace(
                self.keys_avg, rearrange(keys_sum, "c d n -> c n d"), self.decay
            )
            # cluster_size.shape = cj
            cluster_size = laplace_smoothing(
                self.cluster_size, self.key_value_pairs_per_codebook, self.eps
            ) * self.cluster_size.sum(-1, keepdims=True)
            # keys_normalized.shape = self.keys_avg.shape = cjk
            keys_normalized = self.keys_avg / cluster_size[:, :, None]
            self.keys.data.copy_(keys_normalized)
            self.expire_codes_(flatten)
        else:
            if self.count_cluster_size:
                # cluster_size.shape = cj
                cluster_size = self.get_cluster_size(keys_ind)

                # cluster_size = keys_onehot.sum(1)
                self.cluster_size_counter.data.add_(cluster_size)
        # Remember: i = (b * n * topk)
        quantized_keys = rearrange(quantized_keys, "c (b n topk) k -> b c topk n k", b=b, n=n, topk=forward_topk)
        keys_ind = rearrange(keys_ind, "c (b n topk) -> b c topk n", b=b, n=n, topk=forward_topk)[:, :, :, :, None]
        quantized_values = rearrange(quantized_values, "c (b n topk) v -> b c topk n v", b=b, n=n, topk=forward_topk)
        dist_to_fetched_keys = rearrange(
            dist_to_fetched_keys, "c (b n) topk -> b c topk n", b=b, n=n, topk=forward_topk
        )[:, :, :, :, None]
        counts = rearrange(
            counts, "c (b n) topk -> b c topk n", b=b, n=n, topk=forward_topk
        )[:, :, :, :, None]
        if self.return_values_only:
            return (quantized_values,)
        else:
            return quantized_values, quantized_keys, keys_ind, dist_to_fetched_keys, counts

    def freeze_keys(self):
        self.update_keys = False
        self.keys.requires_grad = False
        return self

    def unfreeze_keys(self):
        self.update_keys = True
        self.keys.requires_grad = True
        return self

    def fraction_of_unused_keys(self):
        num_not_used_pairs = (self.cluster_size_counter == 0).sum()
        num_total_pairs = torch.numel(self.cluster_size_counter)
        return num_not_used_pairs / num_total_pairs

    def get_cluster_size(self, keys_ind):
        keys_ind = einops.rearrange(keys_ind, "c (i topk) -> c i topk", topk=self.topk)
        keys_onehot = F.one_hot(keys_ind, self.key_value_pairs_per_codebook)
        cluster_size = keys_onehot.sum(1)
        cluster_size = cluster_size.sum(1)
        return cluster_size


class BottleneckedEncoder(nn.Module):
    VALID_SPLITTING_MODES = {"chunk", "random_projection", "sparse_projection", "learned_projection"}

    def __init__(
            self,
            encoder: nn.Module,
            num_channels: int,
            num_codebooks: int,
            key_value_pairs_per_codebook: int,
            dim_keys: int,
            dim_values: int,
            init_mode: str = "random",
            kmeans_iters: int = 10,
            splitting_mode: str = "random_projection",
            decay: float = 0.95,
            eps: float = 1e-5,
            threshold_ema_dead_code: float = 0.0,
            encoder_is_channel_last: bool = True,
            concat_values_from_all_codebooks: bool = False,
            sample_codebook_temperature: float = 0.0,
            return_values_only: bool = True,
            topk: int = 1,
    ):
        super().__init__()
        # Store attributes
        self.activate_pre_transform = False
        self.transforms = None
        self.num_channels = num_channels
        self.num_codebooks = num_codebooks
        self.key_value_pairs_per_codebook = key_value_pairs_per_codebook
        self.splitting_mode = splitting_mode
        self.dim_keys = dim_keys
        self.dim_values = dim_values
        self.init_mode = init_mode
        self.kmeans_iters = kmeans_iters
        self.decay = decay
        self.eps = eps
        self.topk = topk
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.encoder_is_channel_last = encoder_is_channel_last
        self.concat_values_from_all_codebooks = concat_values_from_all_codebooks
        self.sample_codebook_temperature = sample_codebook_temperature
        self.return_values_only = return_values_only
        # Initialize the modules
        self.encoder = encoder
        self.frozen_encoder = False
        self.frozen_splitter = False
        self.splitter = self._init_splitter()
        self.bottleneck = self._init_bottleneck()

    def _init_bottleneck(self) -> "KeyValueBottleneck":
        return KeyValueBottleneck(
            num_codebooks=self.num_codebooks,
            key_value_pairs_per_codebook=self.key_value_pairs_per_codebook,
            dim_keys=self.dim_keys,
            dim_values=self.dim_values,
            init_mode=self.init_mode,
            kmeans_iters=self.kmeans_iters,
            decay=self.decay,
            eps=self.eps,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            sample_codebook_temperature=self.sample_codebook_temperature,
            return_values_only=self.return_values_only,
            topk=self.topk,
        )

    def _init_splitter(self):
        assert self.splitting_mode in self.VALID_SPLITTING_MODES
        if self.splitting_mode == "chunk":
            return Chunker(
                num_codebooks=self.num_codebooks,
                num_channels=self.num_channels,
                dim_keys=self.dim_keys,
            )
        elif self.splitting_mode == "random_projection":
            return RandomDownProjection(
                num_codebooks=self.num_codebooks,
                num_channels=self.num_channels,
                dim_keys=self.dim_keys,
            )
        elif self.splitting_mode == "learned_projection":
            return LearnedDownProjection(
                num_codebooks=self.num_codebooks,
                num_channels=self.num_channels,
                dim_keys=self.dim_keys,
            )
        else:
            raise ValueError(f"Unknown splitting mode {self.splitting_mode}")

    def _reshape_to_channel_last(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[int]]:
        if not self.encoder_is_channel_last:
            # We'll need to reshape bd... to b...d
            _, _, *spatial_shape = x.shape
            x = rearrange(x, "b d ... -> b (...) d")
        else:
            # spatial_shape = None
            _, *spatial_shape, _ = x.shape
            x = rearrange(x, "b ... d -> b (...) d")
        return x, spatial_shape

    def _undo_reshape(self, x: torch.Tensor, spatial_shape: List[int]) -> torch.Tensor:
        if not self.encoder_is_channel_last:
            assert spatial_shape is not None
            dim_strings = " ".join(list(string.ascii_lowercase[: len(spatial_shape)]))
            dim_kwargs = {
                k: spatial_shape[idx]
                for idx, k in enumerate(string.ascii_lowercase[: len(spatial_shape)])
            }
            if self.concat_values_from_all_codebooks:
                reshaper = (
                    f"batch ({dim_strings}) channels -> batch channels {dim_strings}"
                )
            else:
                reshaper = f"batch codebook topk ({dim_strings}) channels -> batch codebook topk channels {dim_strings}"
            x = rearrange(x, reshaper, **dim_kwargs)
        return x

    def forward_splitter(self, x: torch.Tensor):
        # x.shape is anything that the encoder wants
        if self.frozen_encoder:
            self.encoder.eval()
            with torch.no_grad():
                # x.shape = bd... or b...d
                x_encoder = self.encoder(x)
        else:
            # x.shape = bd... or b...d
            x_encoder = self.encoder(x)
        # x.shape = b(...)d
        x, spatial_shape = self._reshape_to_channel_last(x_encoder)
        # x.shape = bc(...)k
        if self.frozen_splitter:
            with torch.no_grad():
                x = self.splitter(x)
        else:
            x = self.splitter(x)

        return x

    def forward(self, x: torch.Tensor):
        # x.shape is anything that the encoder wants
        if self.activate_pre_transform:
            x = self.transforms(x)
            device = next(self.parameters()).device
            x = x.to(device)
        self.encoder.eval()
        if self.frozen_encoder:
            with torch.inference_mode():
                # x.shape = bd... or b...d
                x_encoder = self.encoder(x)
        else:
            # x.shape = bd... or b...d
            x_encoder = self.encoder(x)
        # x.shape = b(...)d
        x, spatial_shape = self._reshape_to_channel_last(x_encoder)
        # x.shape = bc(...)k
        if self.frozen_splitter:
            with torch.inference_mode():
                x_splitter = self.splitter(x)
        else:
            x_splitter = self.splitter(x)
        # x.shape = bc(...)v
        x, *extra_info = self.bottleneck(x_splitter)
        # x.shape = bc(...)v
        if self.concat_values_from_all_codebooks:
            # x.shape = b(...)(nv)
            x = rearrange(
                x, "b c n v -> b n (c v)", c=self.num_codebooks, v=self.dim_values
            )
        x = self._undo_reshape(x, spatial_shape)
        if extra_info is not None:
            for info_idx in range(len(extra_info)):
                extra_info[info_idx] = self._undo_reshape(extra_info[info_idx], spatial_shape)
            extra_info.append(x_splitter)
            extra_info.append(x_encoder)
        if len(extra_info) == 0:
            return x
        else:
            return_tuple = (x,) + tuple(extra_info)
            return return_tuple

    def freeze_encoder(self) -> "BottleneckedEncoder":
        self.frozen_encoder = True
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        return self

    def freeze_splitter(self) -> "BottleneckedEncoder":
        self.frozen_splitter = True
        self.splitter.eval()
        for p in self.splitter.parameters():
            p.requires_grad = False
        return self

    def unfreeze_splitter(self) -> "BottleneckedEncoder":
        self.frozen_splitter = False
        for p in self.splitter.parameters():
            p.requires_grad = True
        return self

    def freeze_keys(self) -> "BottleneckedEncoder":
        self.bottleneck.freeze_keys()
        return self

    def unfreeze_keys(self):
        self.bottleneck.unfreeze_keys()
        return self

    def reset_cluster_size_counter(self):
        self.bottleneck.reset_cluster_size_counter()

    def deactivate_counts(self):
        self.bottleneck.deactivate_counts()

    def enable_update_keys(self):
        self.bottleneck.enable_update_keys()

    def disable_update_keys(self):
        self.bottleneck.disable_update_keys()

    def activate_counts(self):
        self.bottleneck.activate_counts()

    def fraction_of_unused_keys(self):
        return self.bottleneck.fraction_of_unused_keys()

    def initialize_keys(
            self,
            loader: Iterable,
            epochs: int = 1,
            inputs_extractor: Optional[Callable] = None,
    ):
        device = next(self.parameters()).device
        if inputs_extractor is None:
            inputs_extractor = lambda item: item[0]
        for epoch in range(epochs):
            for sample in tqdm.tqdm(loader):
                inputs = inputs_extractor(sample)
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                self.forward(inputs)
        return self

    def prepare(self, loader: Iterable, epochs: int = 1):
        return (
            self.freeze_encoder().freeze_splitter().initialize_keys(loader, epochs=epochs).freeze_keys()
        )

    def save(self, model_path: str):
        torch.save(
            {
                "bottleneck": self.bottleneck.state_dict(),
                "splitter": self.splitter.state_dict(),
            },
            model_path,
        )

    def load(self, model_path: str):
        state_dict = torch.load(model_path)
        self.bottleneck.load_state_dict(state_dict["bottleneck"])
        self.splitter.load_state_dict(state_dict["splitter"])

    def save_model(self, model_path: str):
        torch.save(self, model_path)


class DinoBottleneckedEncoder(BottleneckedEncoder):
    VALID_BACKBONES = ["dino_resnet50", "dino_vits8"]
    RESNET_LAYER_TO_CHANNELS = {1: 256, 2: 512, 3: 1024, 4: 2048}
    DINO_VITS8_DIM_EMBEDDING = 384

    def __init__(
            self,
            num_codebooks: int,
            key_value_pairs_per_codebook: int,
            dim_keys: Optional[int] = None,
            dim_values: Optional[int] = None,
            init_mode: str = "random",
            kmeans_iters: int = 10,
            backbone: str = "dino_resnet50",
            extracted_layer: int = 3,
            pool_embedding: bool = True,
            splitting_mode: str = "chunk",
            decay: float = 0.95,
            eps: float = 1e-5,
            threshold_ema_dead_code: float = 0.0,
            concat_values_from_all_codebooks: bool = True,
            sample_codebook_temperature: float = 0.0,
            return_values_only: bool = True,
            topk: int = 1,
            input_is_embedding: bool = False,
    ):
        assert backbone in self.VALID_BACKBONES
        self.extracted_layer = extracted_layer
        encoder_is_channel_last = False
        if input_is_embedding:
            if backbone == "dino_resnet50":
                num_channels = self.RESNET_LAYER_TO_CHANNELS[extracted_layer]
            elif backbone == "dino_vits8":
                num_channels = self.DINO_VITS8_DIM_EMBEDDING
            else:
                raise NotImplementedError
            encoder = nn.Identity()
        else:
            pretrained_encoder = torch.hub.load("facebookresearch/dino:main", backbone)
            if backbone == "dino_resnet50":
                encoder_modules = [*pretrained_encoder.children()][: 4 + extracted_layer]
                if pool_embedding:
                    encoder_modules.append(nn.AdaptiveAvgPool2d(1))
                num_channels = self.RESNET_LAYER_TO_CHANNELS[extracted_layer]
                encoder = nn.Sequential(*encoder_modules)
            elif backbone == "dino_vits8":
                encoder = pretrained_encoder
                num_channels = self.DINO_VITS8_DIM_EMBEDDING
            else:
                raise NotImplementedError

        if dim_keys is None:
            dim_keys = num_channels // num_codebooks
            assert dim_keys * num_codebooks == num_channels
        if dim_values is None:
            dim_values = dim_keys

        super().__init__(
            encoder=encoder,
            num_channels=num_channels,
            num_codebooks=num_codebooks,
            key_value_pairs_per_codebook=key_value_pairs_per_codebook,
            dim_keys=dim_keys,
            dim_values=dim_values,
            init_mode=init_mode,
            kmeans_iters=kmeans_iters,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            sample_codebook_temperature=sample_codebook_temperature,
            encoder_is_channel_last=encoder_is_channel_last,
            concat_values_from_all_codebooks=concat_values_from_all_codebooks,
            splitting_mode=splitting_mode,
            return_values_only=return_values_only,
            topk=topk,
        )


class SwavBottleneckedEncoder(BottleneckedEncoder):
    VALID_BACKBONES = ["swav_resnet50", "swav_resnet50w2"]
    RESNET_LAYER_TO_CHANNELS = {1: 256, 2: 512, 3: 1024, 4: 2048}
    RESNETw2_LAYER_TO_CHANNELS = {1: 2 * 256, 2: 2 * 512, 3: 2 * 1024, 4: 2 * 2048}

    def __init__(
            self,
            num_codebooks: int,
            key_value_pairs_per_codebook: int,
            dim_keys: Optional[int] = None,
            dim_values: Optional[int] = None,
            init_mode: str = "random",
            kmeans_iters: int = 10,
            backbone: str = "swav_resnet50",
            extracted_layer: int = 3,
            pool_embedding: bool = True,
            splitting_mode: str = "chunk",
            decay: float = 0.95,
            eps: float = 1e-5,
            threshold_ema_dead_code: float = 0.0,
            concat_values_from_all_codebooks: bool = True,
            sample_codebook_temperature: float = 0.0,
            return_values_only: bool = True,
            topk: int = 1,
            input_is_embedding: bool = False,
    ):
        assert backbone in self.VALID_BACKBONES
        self.extracted_layer = extracted_layer
        encoder_is_channel_last = False

        if input_is_embedding:
            encoder = nn.Identity()
        else:
            if backbone == "swav_resnet50":
                pretrained_encoder = torch.hub.load('facebookresearch/swav:main', 'resnet50')
                encoder_modules = [*pretrained_encoder.children()][: 4 + extracted_layer]
            elif backbone == "swav_resnet50w2":
                pretrained_encoder = torch.hub.load('facebookresearch/swav:main', 'resnet50w2')
                encoder_modules = [*pretrained_encoder.children()][: 5 + extracted_layer]
            else:
                raise NotImplementedError
            if pool_embedding:
                encoder_modules.append(nn.AdaptiveAvgPool2d(1))
            encoder = nn.Sequential(*encoder_modules)
            encoder = SwavModelWrapper(encoder)

        if backbone == "swav_resnet50":
            num_channels = self.RESNET_LAYER_TO_CHANNELS[extracted_layer]
        elif backbone == "swav_resnet50w2":
            num_channels = self.RESNETw2_LAYER_TO_CHANNELS[extracted_layer]
        else:
            raise NotImplementedError

        if dim_keys is None:
            dim_keys = num_channels // num_codebooks
            assert dim_keys * num_codebooks == num_channels
        if dim_values is None:
            dim_values = dim_keys

        super().__init__(
            encoder=encoder,
            num_channels=num_channels,
            num_codebooks=num_codebooks,
            key_value_pairs_per_codebook=key_value_pairs_per_codebook,
            dim_keys=dim_keys,
            dim_values=dim_values,
            init_mode=init_mode,
            kmeans_iters=kmeans_iters,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            sample_codebook_temperature=sample_codebook_temperature,
            encoder_is_channel_last=encoder_is_channel_last,
            concat_values_from_all_codebooks=concat_values_from_all_codebooks,
            splitting_mode=splitting_mode,
            return_values_only=return_values_only,
            topk=topk,
        )


class VICRegBottleneckedEncoder(BottleneckedEncoder):
    VALID_BACKBONES = ["vicreg_resnet50", "vicreg_resnet50w2"]
    RESNET_LAYER_TO_CHANNELS = {1: 256, 2: 512, 3: 1024, 4: 2048}
    RESNETw2_LAYER_TO_CHANNELS = {1: 2*256, 2: 2*512, 3: 2*1024, 4: 2*2048}

    def __init__(
            self,
            num_codebooks: int,
            key_value_pairs_per_codebook: int,
            dim_keys: Optional[int] = None,
            dim_values: Optional[int] = None,
            init_mode: str = "random",
            kmeans_iters: int = 10,
            backbone: str = "swav_resnet50",
            extracted_layer: int = 3,
            pool_embedding: bool = True,
            splitting_mode: str = "chunk",
            decay: float = 0.95,
            eps: float = 1e-5,
            threshold_ema_dead_code: float = 0.0,
            concat_values_from_all_codebooks: bool = True,
            sample_codebook_temperature: float = 0.0,
            return_values_only: bool = True,
            topk: int = 1,
            input_is_embedding: bool = False,
    ):
        assert backbone in self.VALID_BACKBONES
        self.extracted_layer = extracted_layer
        encoder_is_channel_last = False
        if input_is_embedding:
            encoder = nn.Identity()
        else:
            if backbone == "vicreg_resnet50":
                pretrained_encoder = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
                encoder_modules = [*pretrained_encoder.children()][: 5 + extracted_layer]
            elif backbone == "vicreg_resnet50w2":
                pretrained_encoder = torch.hub.load('facebookresearch/vicreg:main', 'resnet50x2')
                encoder_modules = [*pretrained_encoder.children()][: 5 + extracted_layer]
            else:
                raise NotImplementedError
            if pool_embedding:
                encoder_modules.append(nn.AdaptiveAvgPool2d(1))
            encoder = nn.Sequential(*encoder_modules)

        if backbone == "vicreg_resnet50":
            num_channels = self.RESNET_LAYER_TO_CHANNELS[extracted_layer]
        elif backbone == "vicreg_resnet50w2":
            num_channels = self.RESNETw2_LAYER_TO_CHANNELS[extracted_layer]
        else:
            raise NotImplementedError

        if dim_keys is None:
            dim_keys = num_channels // num_codebooks
            assert dim_keys * num_codebooks == num_channels
        if dim_values is None:
            dim_values = dim_keys

        super().__init__(
            encoder=encoder,
            num_channels=num_channels,
            num_codebooks=num_codebooks,
            key_value_pairs_per_codebook=key_value_pairs_per_codebook,
            dim_keys=dim_keys,
            dim_values=dim_values,
            init_mode=init_mode,
            kmeans_iters=kmeans_iters,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            sample_codebook_temperature=sample_codebook_temperature,
            encoder_is_channel_last=encoder_is_channel_last,
            concat_values_from_all_codebooks=concat_values_from_all_codebooks,
            splitting_mode=splitting_mode,
            return_values_only=return_values_only,
            topk=topk,
        )


class SLPretrainedBottleneckedEncoder(BottleneckedEncoder):
    VALID_BACKBONES = ["cifar10_resnet56", "cifar10_resnet22", "cifar100_resnet56", "cifar100_resnet22",
                       "resnet50_imagenet_v2"]
    CIFAR_RESNET_LAYER_TO_CHANNELS = {1: 16, 2: 32, 3: 64}
    IMAGENET_RESNET_LAYER_TO_CHANNELS = {1: 256, 2: 512, 3: 1024, 4: 2048}

    def __init__(
            self,
            num_codebooks: int,
            key_value_pairs_per_codebook: int,
            dim_keys: Optional[int] = None,
            dim_values: Optional[int] = None,
            init_mode: str = "random",
            kmeans_iters: int = 10,
            backbone: str = "cifar10_resnet56",
            extracted_layer: int = 3,
            pool_embedding: bool = True,
            splitting_mode: str = "chunk",
            decay: float = 0.95,
            eps: float = 1e-5,
            threshold_ema_dead_code: float = 0.0,
            concat_values_from_all_codebooks: bool = True,
            sample_codebook_temperature: float = 0.0,
            return_values_only: bool = True,
            topk: int = 1,
            input_is_embedding: bool = False,
    ):
        assert backbone in self.VALID_BACKBONES
        self.extracted_layer = extracted_layer
        encoder_is_channel_last = False
        transforms = None
        if input_is_embedding:
            encoder = nn.Identity()
        else:
            if "cifar10" in backbone:
                pretrained_encoder = torch.hub.load("chenyaofo/pytorch-cifar-models", backbone, pretrained=True)
                encoder_is_channel_last = False
                encoder_modules = [*pretrained_encoder.children()][:3 + extracted_layer]
                if pool_embedding:
                    encoder_modules.append(nn.AdaptiveAvgPool2d(1))
                encoder = nn.Sequential(*encoder_modules)
                transforms = None
            elif "resnet50_imagenet_v2" in backbone:
                from torchvision.models import resnet50, ResNet50_Weights
                encoder_is_channel_last = False
                weights = ResNet50_Weights.IMAGENET1K_V2
                pretrained_encoder = resnet50(weights=weights)
                encoder_modules = [*pretrained_encoder.children()][:4 + extracted_layer]
                if pool_embedding:
                    encoder_modules.append(nn.AdaptiveAvgPool2d(1))
                encoder = nn.Sequential(*encoder_modules)
                transforms = weights.transforms()
            else:
                raise NotImplementedError

        if "cifar10" in backbone:
            num_channels = self.CIFAR_RESNET_LAYER_TO_CHANNELS[extracted_layer]
        elif "resnet50_imagenet_v2" in backbone:
            num_channels = self.IMAGENET_RESNET_LAYER_TO_CHANNELS[extracted_layer]
        else:
            raise NotImplementedError

        if dim_keys is None:
            dim_keys = num_channels // num_codebooks
            assert dim_keys * num_codebooks == num_channels
        if dim_values is None:
            dim_values = dim_keys

        super().__init__(
            encoder=encoder,
            num_channels=num_channels,
            num_codebooks=num_codebooks,
            key_value_pairs_per_codebook=key_value_pairs_per_codebook,
            dim_keys=dim_keys,
            dim_values=dim_values,
            init_mode=init_mode,
            kmeans_iters=kmeans_iters,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            sample_codebook_temperature=sample_codebook_temperature,
            encoder_is_channel_last=encoder_is_channel_last,
            concat_values_from_all_codebooks=concat_values_from_all_codebooks,
            splitting_mode=splitting_mode,
            return_values_only=return_values_only,
            topk=topk,
        )
        self.transforms = transforms


class CLIPBottleneckedEncoder(BottleneckedEncoder):
    VALID_BACKBONES = ["clip_vit_b32"]
    CLIP_VIT_B32_DIM_EMBEDDING = 512

    def __init__(
            self,
            num_codebooks: int,
            key_value_pairs_per_codebook: int,
            dim_keys: Optional[int] = None,
            dim_values: Optional[int] = None,
            init_mode: str = "random",
            kmeans_iters: int = 10,
            backbone: str = "clip_vit_b32",
            splitting_mode: str = "chunk",
            decay: float = 0.95,
            eps: float = 1e-5,
            threshold_ema_dead_code: float = 0.0,
            concat_values_from_all_codebooks: bool = True,
            sample_codebook_temperature: float = 0.0,
            return_values_only: bool = True,
            topk: int = 1,
            input_is_embedding: bool = False,
            **kwargs
    ):
        assert backbone in self.VALID_BACKBONES
        self.kwargs = kwargs
        encoder_is_channel_last = False
        if input_is_embedding:
            encoder = nn.Identity()
            preprocess = None
        else:
            if backbone == "clip_vit_b32":
                import clip
                model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
                encoder = CLIPModelWrapper(model)
            else:
                raise NotImplementedError

        if backbone == "clip_vit_b32":
            num_channels = self.CLIP_VIT_B32_DIM_EMBEDDING
        else:
            raise NotImplementedError

        if dim_keys is None:
            dim_keys = num_channels // num_codebooks
            assert dim_keys * num_codebooks == num_channels
        if dim_values is None:
            dim_values = dim_keys

        super().__init__(
            encoder=encoder,
            num_channels=num_channels,
            num_codebooks=num_codebooks,
            key_value_pairs_per_codebook=key_value_pairs_per_codebook,
            dim_keys=dim_keys,
            dim_values=dim_values,
            init_mode=init_mode,
            kmeans_iters=kmeans_iters,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            sample_codebook_temperature=sample_codebook_temperature,
            encoder_is_channel_last=encoder_is_channel_last,
            concat_values_from_all_codebooks=concat_values_from_all_codebooks,
            splitting_mode=splitting_mode,
            return_values_only=return_values_only,
            topk=topk,
        )
        self.transforms = preprocess


class ConvMixerBottleneckedEncoder(BottleneckedEncoder):
    VALID_BACKBONES = ["convmixer"]
    CONVMIXER_DIM_EMBEDDING = 256

    def __init__(
            self,
            num_codebooks: int,
            key_value_pairs_per_codebook: int,
            dim_keys: Optional[int] = None,
            dim_values: Optional[int] = None,
            init_mode: str = "random",
            kmeans_iters: int = 10,
            backbone: str = "convmixer",
            splitting_mode: str = "chunk",
            decay: float = 0.95,
            eps: float = 1e-5,
            threshold_ema_dead_code: float = 0.0,
            concat_values_from_all_codebooks: bool = True,
            sample_codebook_temperature: float = 0.0,
            return_values_only: bool = True,
            topk: int = 1,
            input_is_embedding: bool = False,
            **kwargs
    ):
        assert backbone in self.VALID_BACKBONES
        self.kwargs = kwargs
        encoder_is_channel_last = False
        if input_is_embedding:
            encoder = nn.Identity()
            preprocess = None
        else:
            raise NotImplementedError

        if backbone == "convmixer":
            num_channels = self.CONVMIXER_DIM_EMBEDDING
        else:
            raise NotImplementedError

        if dim_keys is None:
            dim_keys = num_channels // num_codebooks
            assert dim_keys * num_codebooks == num_channels
        if dim_values is None:
            dim_values = dim_keys

        super().__init__(
            encoder=encoder,
            num_channels=num_channels,
            num_codebooks=num_codebooks,
            key_value_pairs_per_codebook=key_value_pairs_per_codebook,
            dim_keys=dim_keys,
            dim_values=dim_values,
            init_mode=init_mode,
            kmeans_iters=kmeans_iters,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            sample_codebook_temperature=sample_codebook_temperature,
            encoder_is_channel_last=encoder_is_channel_last,
            concat_values_from_all_codebooks=concat_values_from_all_codebooks,
            splitting_mode=splitting_mode,
            return_values_only=return_values_only,
            topk=topk,
        )
        self.transforms = preprocess
