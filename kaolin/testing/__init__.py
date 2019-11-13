"""
Testing specific utils
"""

import torch

# Borrowed from kornia
# https://github.com/arraiyopensource/kornia
# https://github.com/kornia/kornia/blob/master/kornia/testing/__init__.py
def tensor_to_gradcheck_var(tensor, dtype=torch.float64, requires_grad=True):
    """Makes input tensors gradcheck-compatible (i.e., float64, and
       requires_grad = True).
    """

    assert torch.is_tensor(tensor), type(tensor)
    return tensor.requires_grad_(requires_grad).type(dtype)
