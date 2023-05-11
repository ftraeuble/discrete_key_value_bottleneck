# Copied from Phil Wang's repository.
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import einops
import torch


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_()
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1, topk=1):
    if topk == 0:
        if temperature == 0:
            return t.argmax(dim=dim)

        return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)
    elif topk == 1:
        if temperature == 0:
            return t.argmax(dim=dim)[..., None]

        return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)[..., None]
    else:
        if temperature == 0:
            result = t.topk(k=topk, largest=True, dim=dim).indices
            return result

        result = ((t / temperature) + gumbel_noise(t)).topk(k=topk, largest=True, dim=dim).indices
        return result


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories, eps=1e-5):
    # x.shape = j or cj, where c is the number of codebooks
    # and j is the number of keys.
    assert x.ndim in [1, 2]
    return (x + eps) / (x.sum(-1, keepdims=True) + n_categories * eps)


def sample_vectors(samples, num):
    # samples.shape = cik
    # num = j
    assert samples.ndim == 3
    device = samples.device
    # num_samples = i
    num_codebooks = samples.shape[0]
    num_samples = samples.shape[1]
    # indices.shape = cj
    indices = torch.randint(
        0,
        num_samples,
        (
            num_codebooks,
            num,
        ),
        device=device,
    )
    # samples[indices].shape = cjk
    indices = einops.repeat(indices, "c j -> c j k", k=samples.shape[2])
    return torch.gather(samples, dim=1, index=indices)


def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


def kmeans(samples, num_clusters, num_iters=10):
    # samples.shape = cik
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device
    num_codebooks = samples.shape[0]
    # means.shape = cjk
    means = sample_vectors(samples, num_clusters)
    bins = None
    for _ in range(num_iters):
        # diffs.shape = cijk
        diffs = einops.rearrange(samples, "c i k -> c i () k") - einops.rearrange(
            means, "c j k -> c () j k"
        )
        # diffs.shape = cij
        dists = -(diffs ** 2).sum(dim=-1)
        # buckets.shape = ci
        buckets = dists.max(dim=-1).indices
        # bins.shape = cj
        bins = batched_bincount(buckets, dim=-1, max_value=num_clusters)
        # zero_mask.shape = cj
        zero_mask = bins == 0
        # bins_min_clamped.shape = cj
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        # new_means.shape = cjk
        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(1, einops.repeat(buckets, "c i -> c i k", k=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        # means.shape = cjk
        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins
