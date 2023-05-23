import torch
import torchvision.datasets
from torch import nn
import os
from key_value_bottleneck.core import (
    KeyValueBottleneck,
    BottleneckedEncoder,
    DinoBottleneckedEncoder,
    SLPretrainedBottleneckedEncoder,
    CLIPBottleneckedEncoder
)

def forward_and_backward(bottlenecked_encoder, dataloader_cifar10):
    bottlenecked_encoder.reset_cluster_size_counter()
    bottlenecked_encoder.disable_update_keys()
    _ = bottlenecked_encoder(next(iter(dataloader_cifar10))[0])
    output = bottlenecked_encoder(next(iter(dataloader_cifar10))[0])
    loss = torch.mean(output[0])
    loss.backward()
    bottlenecked_encoder.deactivate_counts()
    output = bottlenecked_encoder(next(iter(dataloader_cifar10))[0])


def test_bottleneck():
    batch_size = 256
    num_books = 64
    dim_keys = 32
    dim_values = 64
    num_pairs = 200
    topk = 1
    num_channels = 3

    bottleneck = KeyValueBottleneck(
        num_codebooks=num_books,
        key_value_pairs_per_codebook=num_pairs,
        dim_keys=dim_keys,
        dim_values=dim_values,
        return_values_only=False,
    )
    shape = (num_channels,)
    x = torch.randn(batch_size, num_books, *shape, dim_keys)
    quantized_values, quantized_keys, keys_idx, dists, counts = bottleneck(x)

    assert quantized_values.shape == (batch_size, num_books, topk, num_channels, dim_values)
    assert quantized_keys.shape == (batch_size, num_books, topk, num_channels, dim_keys)
    assert keys_idx.shape == (batch_size, num_books, topk, num_channels, 1)
    assert dists.shape == (batch_size, num_books, topk, num_channels, 1)
    assert counts.shape == (batch_size, num_books, topk, num_channels, 1)


def test_bottlenecked_encoder():
    # Step 1: Load the encoder
    encoder = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
    pretrain_layer = 3
    encoder = nn.Sequential(*([*encoder.children()][: 4 + pretrain_layer]))

    # Step 2: Apply the key value bottleneck
    # Option 1:
    # BottleneckedEncoder is a wrapper around encoder. It instantiates the key value bottleneck and
    # the codebook without the user having to worry about it.
    batch_size = 128
    num_codebooks = 16
    num_channels = 1024
    dataset_cifar10 = torchvision.datasets.CIFAR10(
        root="./data",
        transform=torchvision.transforms.ToTensor(),
        train=True,
        download=True,
    )
    dataloader_cifar10 = torch.utils.data.DataLoader(
        dataset_cifar10, batch_size=batch_size, shuffle=True
    )
    bottlenecked_encoder = BottleneckedEncoder(
        encoder,
        encoder_is_channel_last=False,
        num_channels=num_channels,
        num_codebooks=num_codebooks,
        key_value_pairs_per_codebook=100,
        splitting_mode="chunk",
        dim_keys=num_channels // num_codebooks,
        dim_values=2,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=0.2,
    )
    output = bottlenecked_encoder.prepare(loader=dataloader_cifar10, epochs=0)


def test_dino_bottlenecked_encoder():
    # DinoBottleneckedEncoder is a wrapper around a dino pretrained encoder.
    # It instantiates the key value bottleneck and
    # the codebook without the user having to worry about it.

    batch_size = 128
    num_codebooks = 8
    dataset_cifar10 = torchvision.datasets.CIFAR10(
        root="./data",
        transform=torchvision.transforms.ToTensor(),
        train=True,
        download=True,
    )
    dataloader_cifar10 = torch.utils.data.DataLoader(
        dataset_cifar10, batch_size=batch_size, shuffle=True
    )
    bottlenecked_encoder = DinoBottleneckedEncoder(
        extracted_layer=3,
        pool_embedding=True,
        backbone="dino_resnet50",
        num_codebooks=num_codebooks,
        key_value_pairs_per_codebook=100,
        concat_values_from_all_codebooks=False,
        init_mode="kmeans",
        kmeans_iters=10,
        splitting_mode="learned_projection",
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=0.2,
        return_values_only=False,
        topk=4
    )

    forward_and_backward(bottlenecked_encoder, dataloader_cifar10)


def test_dino_vits_bottlenecked_encoder():
    # DinoBottleneckedEncoder is a wrapper around a dino pretrained encoder.
    # It instantiates the key value bottleneck and
    # the codebook without the user having to worry about it.

    batch_size = 128
    num_codebooks = 20
    dataset_cifar10 = torchvision.datasets.CIFAR10(
        root="./data",
        transform=torchvision.transforms.ToTensor(),
        train=True,
        download=True,
    )
    dataloader_cifar10 = torch.utils.data.DataLoader(
        dataset_cifar10, batch_size=batch_size, shuffle=True
    )
    bottlenecked_encoder = DinoBottleneckedEncoder(
        extracted_layer=3,
        pool_embedding=True,
        dim_keys=256,
        dim_values=256,
        num_codebooks=num_codebooks,
        key_value_pairs_per_codebook=100,
        backbone="dino_vits8",
        concat_values_from_all_codebooks=False,
        init_mode="kmeans",
        kmeans_iters=10,
        splitting_mode="random_projection",
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=0.2,
        return_values_only=False,
    )

    model_path = "bottlenecked_encoder_dino_vits.pt"
    if os.path.exists(model_path):
        print("Loading model from:", model_path)
        bottlenecked_encoder.load(model_path)
    else:
        print("Initializing model")
        bottlenecked_encoder.prepare(loader=dataloader_cifar10, epochs=0)
        bottlenecked_encoder.save(model_path=model_path)

    forward_and_backward(bottlenecked_encoder, dataloader_cifar10)


def test_clip_bottlenecked_encoder():
    # DinoBottleneckedEncoder is a wrapper around a dino pretrained encoder.
    # It instantiates the key value bottleneck and
    # the codebook without the user having to worry about it.

    batch_size = 128
    num_codebooks = 20
    dataset_cifar10 = torchvision.datasets.CIFAR10(
        root="./data",
        transform=torchvision.transforms.ToTensor(),
        train=True,
        download=True,
    )
    dataloader_cifar10 = torch.utils.data.DataLoader(
        dataset_cifar10, batch_size=batch_size, shuffle=True
    )
    bottlenecked_encoder = CLIPBottleneckedEncoder(
        extracted_layer=3,
        pool_embedding=True,
        dim_keys=256,
        dim_values=256,
        num_codebooks=num_codebooks,
        key_value_pairs_per_codebook=100,
        backbone="clip_vit_b32",
        concat_values_from_all_codebooks=False,
        init_mode="kmeans",
        kmeans_iters=10,
        splitting_mode="random_projection",
        decay=0.8,
        eps=1e-5,
        topk=4,
        threshold_ema_dead_code=0.2,
        return_values_only=False,
    )

    if bottlenecked_encoder.transforms is not None:
        dataloader_cifar10.dataset.transform = bottlenecked_encoder.transforms

    forward_and_backward(bottlenecked_encoder, dataloader_cifar10)


def test_sl_resnet_bottlenecked_encoder():
    # DinoBottleneckedEncoder is a wrapper around a dino pretrained encoder.
    # It instantiates the key value bottleneck and
    # the codebook without the user having to worry about it.

    batch_size = 128
    num_codebooks = 20
    dataset_cifar10 = torchvision.datasets.CIFAR10(
        root="./data",
        transform=torchvision.transforms.ToTensor(),
        train=True,
        download=True,
    )
    dataloader_cifar10 = torch.utils.data.DataLoader(
        dataset_cifar10, batch_size=batch_size, shuffle=True
    )
    bottlenecked_encoder = SLPretrainedBottleneckedEncoder(
        extracted_layer=3,
        pool_embedding=True,
        dim_keys=256,
        dim_values=256,
        num_codebooks=num_codebooks,
        key_value_pairs_per_codebook=100,
        backbone="resnet50_imagenet_v2",
        concat_values_from_all_codebooks=False,
        init_mode="kmeans",
        kmeans_iters=10,
        splitting_mode="random_projection",
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=0.2,
        return_values_only=False,
        topk=3
    )

    forward_and_backward(bottlenecked_encoder, dataloader_cifar10)

