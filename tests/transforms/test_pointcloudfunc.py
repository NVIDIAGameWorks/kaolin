import pytest

import torch
from torch.testing import assert_allclose

import kaolin as kal

def test_realign(device='cpu'):
    src = torch.randn(4, 3).to(device)
    tgt = torch.arange(4).expand(3, 4).t().to(device, dtype=torch.float)
    src_ = kal.transforms.pointcloudfunc.realign(src, tgt)
