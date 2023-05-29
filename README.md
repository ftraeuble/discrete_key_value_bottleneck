## Discrete Key Value Bottleneck - PyTorch Module

> [**Discrete Key-Value Bottleneck**](https://arxiv.org/abs/2207.11240)
> *Frederik Träuble, Anirudh Goyal, Nasim Rahaman, Michael Mozer, Kenji Kawaguchi, Yoshua Bengio, Bernhard Schölkopf*. ICML 2023.

See the `experiments_discrete_key_value_bottleneck` 
[repository](https://github.com/ftraeuble/experiments_discrete_key_value_bottleneck) to run experiments from the paper.

### Install as pip module
```bash
conda create -n kvb python=3.10.6
conda activate kvb 
git clone git@github.com:ftraeuble/discrete_key_value_bottleneck.git
cd discrete_key_value_bottleneck
pip install .
```

### How to use the model?

A key-value-bottleneck, that can be used as plug-and-play nn.Module can be easily instantiated
```python
import torch
from key_value_bottleneck.core import KeyValueBottleneck

batch_size = 256
num_books = 64
dim_keys = 32
dim_values = 16
num_pairs = 200
num_channels = 10

bottleneck = KeyValueBottleneck(
    num_codebooks=num_books,
    key_value_pairs_per_codebook=num_pairs,
    dim_keys=dim_keys,
    dim_values=dim_values,
    topk=1,
    return_values_only=False,
)
x = torch.randn(batch_size, num_books, num_channels, dim_keys)
quantized_values, quantized_keys, keys_idx, dists, counts = bottleneck(x)

# quantized_values.shape = (batch_size, num_books, topk, num_channels, dim_values) torch.Size([256, 64, 1, 10, 16])
# quantized_keys.shape = (batch_size, num_books, topk, num_channels, dim_keys) torch.Size([256, 64, 1, 10, 32])
# keys_idx.shape = (batch_size, num_books, topk, num_channels, 1) torch.Size([256, 64, 1, 10, 1])
# dists.shape = (batch_size, num_books, topk, num_channels, 1) torch.Size([256, 64, 1, 10, 1])
# counts.shape = (batch_size, num_books, topk, num_channels, 1) torch.Size([256, 64, 1, 10, 1])
```

In addition, the codebase contains some Wrappers that wrap the bottleneck around an encoder and 
provide some basic features for initializing and (un-)freezing various components of the model:

```python
import torch
import torch.nn as nn
from key_value_bottleneck.core import BottleneckedEncoder, DinoBottleneckedEncoder

batch_size = 256
num_books = 64
dim_keys = 16
dim_values = 10
num_pairs = 200
init_epochs = 10

# Step 1 (Option A): Wrap custom encoder with bottleneck
encoder = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
pretrain_layer = 3
encoder = nn.Sequential(*([*encoder.children()][:4+pretrain_layer]), nn.AdaptiveAvgPool2d(1))

bottlenecked_encoder = BottleneckedEncoder(
    encoder,
    encoder_is_channel_last=False,
    num_channels=1024,
    num_codebooks=num_books,
    key_value_pairs_per_codebook=num_pairs,
    splitting_mode="random_projection",
    dim_keys=dim_keys,
    dim_values=dim_values,
    decay=0.95,
    eps=1e-5,
    threshold_ema_dead_code=0.2,
    return_values_only=False,
)

# Step 1 (Option B): Use predefined DinoBottleneckedEncoder
bottlenecked_encoder = DinoBottleneckedEncoder(
    extracted_layer=3,
    pool_embedding=True,
    num_codebooks=num_books,
    key_value_pairs_per_codebook=num_pairs,
    concat_values_from_all_codebooks=False,
    init_mode="kmeans",
    kmeans_iters=10,
    splitting_mode="random_projection",
    dim_keys=dim_keys,
    dim_values=dim_values,
    decay=0.95,
    eps=1e-5,
    threshold_ema_dead_code=0.2,
    return_values_only=False,
    )

# Step 2: Prepare Encoder and Bottleneck for training under distribution shifts
dataloader_for_initialization = DataLoader(...) # Your dataloader
bottlenecked_encoder = bottlenecked_encoder.prepare(loader=dataloader_for_initialization, epochs=init_epochs)

# This step is equivalent to
# bottlenecked_encoder.freeze_encoder()
# bottlenecked_encoder.initialize_keys(loader, epochs=init_epochs)
# bottlenecked_encoder.freeze_keys()

# Step 3: Construct full model
decoder = Decoder(...) # Your decoder
model = FullModel(bottlenecked_encoder, decoder, ...) # A wrapper that combines bottlenecked_encoder and decoder

# Step 4 (Option I): Train in iid setting
dataloader_iid_data = DataLoader(...) # Your dataloader
train(model, dataloader_iid_data, ...) # Your training loop

# Step 4 (Option II): Train in ood setting
dataloader_with_distribution_shift = DataLoader(...) # Your dataloader
freeze_decoder_weights(model) # A method that freezes your decoder weights
train(model, dataloader_with_distribution_shift, ...)
```


### References

Parts of the code are based on the following repositories:

- [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)

Supported pretrained Models are taken from:
- 
- [SwAV models](https://github.com/facebookresearch/swav)
- [VICReg models](https://github.com/facebookresearch/vicreg)
- [DINO models](https://github.com/facebookresearch/dino)
- [CLIP models](https://github.com/openai/CLIP)
- [CIFAR10-pretrained models](https://github.com/chenyaofo/pytorch-cifar-models)
- [ImageNet-pretrained models](https://pytorch.org/vision/stable/models.html)

If you find this code useful, please cite the following paper:

```
@article{trauble2023discrete,
  title={Discrete Key-Value Bottleneck},
  author={Tr{\"a}uble, Frederik and Goyal, Anirudh and Rahaman, Nasim and Mozer, Michael and Kawaguchi, Kenji and Bengio, Yoshua and Sch{\"o}lkopf, Bernhard},
  journal={International Conference on Machine Learning},
  year={2023}
}
```
