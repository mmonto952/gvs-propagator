from numpy.typing import NDArray


try:
    import torch
    ArrayLike = NDArray | torch.Tensor
except ImportError:
    ArrayLike = NDArray
