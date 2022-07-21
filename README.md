# Gumbel-Sigmoid
This repository contains a PyTorch implementation of the Gumbel-Sigmoid distribution.

The code is adapted from the official PyTorch implementation of the Gumbel-Softmax distribution ([link](https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax)).

### Example

```python
In  [1]: import torch

In  [2]: from gumbel_sigmoid import gumbel_sigmoid

In  [3]: x = torch.rand(3,2)

In  [4]: gumbel_sigmoid(x, hard=False)
Out [4]: tensor([[0.3771, 0.4769],
                 [0.4073, 0.9103],
                 [0.9636, 0.5924]])
                
In  [5]: gumbel_sigmoid(x, hard=True)
Out [5]: tensor([[0., 0.],
                 [0., 1.],
                 [1., 1.]])

```
