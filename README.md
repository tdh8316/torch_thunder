# Overview

Train PyTorch models like thunder.

## Installation

```bash
$ conda create -n torch_thunder python=3.12
$ activate torch_thunder
(torch_thunder)$ python -m pip install git+https://github.com/tdh8316/torch_thunder/
```

## Usage

```python
import torch_thunder as tt

model = ...
train_loader = ...
val_loader = ...
optimizer = ...

n_epochs = 10

tt.thunder_train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    n_epochs=n_epochs,
)
```
