## Discrete Key Value Codes - PyTorch Module

### Install as pip module
```console
cd KeyValueBottleneck
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
dim_values = 64
num_pairs = 200

bottleneck = KeyValueBottleneck(
    num_codebooks=num_books,
    key_value_pairs_per_codebook=num_pairs,
    dim_keys=dim_keys,
    dim_values=dim_values,
    return_values_only=False,
)
shape = (4, 8, 16)
x = torch.randn(batch_size, num_books, *shape, dim_keys)
quantized_values, quantized_keys, keys_ind, dists = bottleneck(x)
```

In addition, the codebase contains some Wrappers that wrap the bottleneck around an encoder and provide some base features
for initializing and (un-)freezing various components of the model:

```python
import torch
import torch.nn as nn
from key_value_bottleneck.core import BottleneckedEncoder, DinoBottleneckedEncoder

batch_size = 128
num_codebooks = 16
num_channels = 1024

# Step 1 (Option A): Wrap custom encoder with bottleneck
encoder = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
pretrain_layer = 3
encoder = nn.Sequential(*([*encoder.children()][:4+pretrain_layer]))

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

# Step 1 (Option B): Use predefined DinoBottleneckedEncoder
bottlenecked_encoder = DinoBottleneckedEncoder(
    extracted_layer=3,
    pool_embedding=True,
    num_codebooks=num_codebooks,
    key_value_pairs_per_codebook=100,
    concat_values_from_all_codebooks=False,
    init_mode="kmeans",
    kmeans_iters=10,
    splitting_mode="chunk",
    decay=0.8,
    eps=1e-5,
    threshold_ema_dead_code=0.2,
    return_values_only=False,
    )

# Step 2: Prepare Encoder and Bottleneck for training under distribution shifts
dataloader_for_initialization = get_your_dataloader()
bottlenecked_encoder = bottlenecked_encoder.prepare(loader=dataloader_for_initialization, epochs=0)

# This step is equivalent to
# bottlenecked_encoder.freeze_encoder()
# bottlenecked_encoder.initialize_keys(loader, epochs=epochs)
# bottlenecked_encoder.freeze_keys()

# Step 3: Construct full model
decoder = load_your_favorite_decoder()
model = nn.Sequential(bottlenecked_encoder, decoder)

# Step 4 (Option A): Train in iid setting
dataloader_iid_data = get_your_dataloader()
train(model, dataloader_iid_data, args)

# Step 4 (Option B): Train in ood setting
dataloader_with_distribution_shift = get_your_dataloader()
freeze_decoder_weights(model)
train(model, dataloader_with_distribution_shift, args)
```
