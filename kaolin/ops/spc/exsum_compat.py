# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Backward-compatibility shim for the SPC ``exsum`` (a.k.a. ``prefix_sum``) layout.

Two layouts exist:

* **current** -- ``len(exsum) == num_bytes`` (per octree). ``exsum[i]`` is the
  *inclusive* running sum of set-bit counts of octree bytes ``[0..i]``. The base
  offset of node ``ord`` is ``0 if ord == 0 else exsum[ord - 1]``.
* **legacy** (pre-redesign, deprecated) -- ``len(exsum) == num_bytes + batch_size``.
  Each octree's block is prefixed with a leading ``0`` (so the block is an
  *exclusive* sum with a trailing total). The base offset of node ``ord`` is
  ``exsum[ord]``.

The two carry identical information; per octree ``legacy == cat([0], current)``.
This module detects which layout a tensor is in (purely from its length relative
to the octree byte count and batch size) and converts between them, emitting a
``DeprecationWarning`` when the legacy layout is encountered.

Kept free of any ``kaolin`` imports (only ``torch``/``warnings``) so it can be
imported from anywhere in the package without risking an import cycle.
"""

import warnings

import torch

__all__ = [
    'octree_byte_lengths_from_pyramid',
    'ensure_current_exsum',
    'current_to_legacy',
]

_DEPRECATION_MSG = (
    "{caller} received a legacy `exsum`/`prefix_sum` of length "
    "(num_bytes + batch_size). The exsum convention changed to length num_bytes "
    "(exsum[i] is now the inclusive bit-count sum through byte i). Legacy support "
    "is deprecated and will be removed in a future release; regenerate it with "
    "kaolin.ops.spc.scan_octrees (which now returns the current layout by default)."
)


def octree_byte_lengths_from_pyramid(pyramids):
    r"""Recover the per-octree byte count (number of octree nodes) from a pyramid.

    The number of octree bytes for batch ``b`` equals the cumulative point count
    at ``b``'s ``max_level`` (i.e. all nodes that own children), which is
    ``pyramids[b, 1, max_level]``. ``max_level`` is the highest level with a
    non-zero point count.

    Args:
        pyramids (torch.IntTensor): of shape :math:`(\text{batch_size}, 2,
            \text{max_level} + 2)`, as returned by :func:`scan_octrees`.

    Returns:
        (torch.LongTensor): per-octree byte counts, of shape
        :math:`(\text{batch_size})`, on CPU.
    """
    counts = pyramids[:, 0, :]
    cumsum = pyramids[:, 1, :]
    # number of populated levels - 1 == max_level (every octree has level 0)
    max_level = (counts != 0).sum(dim=1).clamp(min=1) - 1
    byte_lengths = cumsum.gather(1, max_level.unsqueeze(1)).squeeze(1)
    return byte_lengths.to(dtype=torch.long, device='cpu')


def _legacy_to_current(exsum, octree_lengths):
    """Drop the leading ``0`` of each per-octree block."""
    lengths = octree_lengths.to(device=exsum.device, dtype=torch.long)
    if lengths.numel() == 1:
        return exsum[1:].contiguous()
    block_sizes = lengths + 1  # legacy block is num_bytes + 1
    block_starts = torch.zeros(lengths.numel(), dtype=torch.long, device=exsum.device)
    block_starts[1:] = torch.cumsum(block_sizes, dim=0)[:-1]
    keep = torch.ones(exsum.shape[0], dtype=torch.bool, device=exsum.device)
    keep[block_starts] = False  # the leading zero of each block
    return exsum[keep].contiguous()


def current_to_legacy(exsum, octree_lengths):
    r"""Convert a current-layout exsum to the deprecated legacy layout.

    Prepends a leading ``0`` to each per-octree block, yielding total length
    ``num_bytes + batch_size``.

    Args:
        exsum (torch.IntTensor): current-layout exsum of length ``num_bytes``.
        octree_lengths (torch.Tensor): per-octree byte counts.

    Returns:
        (torch.IntTensor): legacy-layout exsum of length
        ``num_bytes + batch_size``, on the same device as ``exsum``.
    """
    lengths = octree_lengths.to(device=exsum.device, dtype=torch.long)
    parts = []
    offset = 0
    for length in lengths.tolist():
        parts.append(exsum.new_zeros(1))
        parts.append(exsum[offset:offset + length])
        offset += length
    return torch.cat(parts).contiguous()


def ensure_current_exsum(exsum, octree_lengths, caller):
    r"""Return ``exsum`` in the current layout, converting + warning if legacy.

    Args:
        exsum (torch.IntTensor): exsum in either the current or legacy layout.
        octree_lengths (torch.Tensor): per-octree byte counts (1D). Its sum is
            the total octree byte count and its element count is the batch size.
        caller (str): name of the calling function, used in the warning message.

    Returns:
        (torch.IntTensor): exsum guaranteed to be in the current layout.
    """
    num_bytes = int(octree_lengths.sum().item())
    batch_size = octree_lengths.numel()
    n = exsum.shape[0]
    if n == num_bytes:
        return exsum
    if n == num_bytes + batch_size:
        warnings.warn(_DEPRECATION_MSG.format(caller=caller),
                      DeprecationWarning, stacklevel=3)
        return _legacy_to_current(exsum, octree_lengths)
    raise ValueError(
        f"{caller}: exsum length {n} is incompatible with an octree of "
        f"{num_bytes} bytes and batch size {batch_size}; expected {num_bytes} "
        f"(current) or {num_bytes + batch_size} (legacy).")
