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
import logging
import os

import torch

from kaolin.io.usd.utils import create_stage
from kaolin.io.usd.gaussians import add_gaussiancloud, import_gaussianclouds
from kaolin.io.usd.subset import add_subset, import_subsets
from kaolin.rep import GaussianSplatModel
from .input_abstraction import GaussianSplatInput
from .segment import DisjointSegmentation

_GAUSSIAN_SCENE_PATH = '/World/Gaussians/gaussian_0'

logger = logging.getLogger(__name__)


def _strip_common_usd_prefix(names: list) -> list:
    """Strip the longest common USD path prefix from all names.

    Splits each name on '/' and finds how many leading components are shared
    across all names. The stripped result is the remaining components joined
    with '/'. If stripping would yield an empty string (a name equals the
    common ancestor), uses the last path component of that name instead.
    """
    if not names:
        return names
    parts = [n.lstrip('/').split('/') for n in names]
    common_depth = 0
    for i in range(min(len(p) for p in parts)):
        if all(p[i] == parts[0][i] for p in parts):
            common_depth = i + 1
        else:
            break
    result = []
    for p in parts:
        remaining = p[common_depth:]
        result.append('/'.join(remaining) if remaining else p[-1])
    return result


def export_scene_as_usd(
    cloud: GaussianSplatInput,
    segmentation: DisjointSegmentation,
    file_path: str,
) -> None:
    """Export the full gaussian cloud and all segment labels to a USD file.

    The file is overwritten if it already exists. Raises on any I/O or USD error.

    Args:
        cloud: the scene's gaussian splat (original, unmodified opacities are used).
        segmentation: disjoint segmentation; all segments including background are written
            as GeomSubset children of the gaussian prim with family_name='part'.
        file_path: absolute server-side path to the output USD file (*.usd or *.usda).
            No path restriction is enforced; callers exposing this over a network should add their own allowlist check.
    """
    file_path = os.path.abspath(file_path)
    if segmentation is None:
        raise ValueError("segmentation must not be None")
    gm = cloud.gsmodel
    stage = create_stage(file_path)
    try:
        add_gaussiancloud(
            stage,
            _GAUSSIAN_SCENE_PATH,
            positions=gm.positions,
            orientations=gm.orientations,
            scales=gm.scales,
            opacities=cloud._orig_opacities,
            sh_coeff=gm.sh_coeff,
        )
        for seg_name, segment in segmentation.segments.items():
            indices = segment.mask.nonzero(as_tuple=False).squeeze(1).long()
            add_subset(stage, _GAUSSIAN_SCENE_PATH, seg_name, indices, family_name='part')
        stage.Save()
    finally:
        del stage


def load_scene_from_usd(fname: str, device: str):
    """Load a gaussian splat scene and its segmentation from a USD file.

    Each prim becomes a segment (named by its USD leaf component). If a prim has
    GeomSubset children with family_name='part', those subsets become the segments
    instead. Uncovered gaussians within a prim (if subsets don't cover it fully)
    form an additional segment named after the prim leaf.

    Segment names have their longest common USD path prefix stripped.

    Args:
        fname: path to the USD file (*.usd or *.usda).
        device: torch device string to move the cloud and masks to.

    Returns:
        (GaussianSplatInput, DisjointSegmentation)

    Raises:
        ValueError: if no ParticleField3DGaussianSplat prims are found.
    """
    clouds = import_gaussianclouds(fname)
    if not clouds:
        raise ValueError(f"No ParticleField3DGaussianSplat prims found in {fname}")
    scene_paths = list(clouds.keys())

    merged = GaussianSplatModel.cat(list(clouds.values())).to(device)
    n_total = len(merged)

    prim_sizes = [len(clouds[p]) for p in scene_paths]
    offsets = []
    running = 0
    for sz in prim_sizes:
        offsets.append(running)
        running += sz

    # Pass 1: collect (raw_name, bool_mask) without knowing the common prefix yet
    segments_data = []  # list of [raw_name, bool_mask]
    for prim_path, offset, prim_size in zip(scene_paths, offsets, prim_sizes):
        subsets = import_subsets(fname, prim_path, family_name='part')

        if subsets:
            prim_covered = torch.zeros(n_total, dtype=torch.bool)
            for usd_path, info in subsets.items():
                indices = info['indices']
                valid = (indices >= 0) & (indices < prim_size)
                if not valid.all():
                    logger.warning(
                        "Subset '%s' has %d out-of-range indices for prim '%s' (size %d); skipping them.",
                        usd_path, int((~valid).sum()), prim_path, prim_size
                    )
                    indices = indices[valid]
                global_indices = indices.long() + offset
                mask = torch.zeros(n_total, dtype=torch.bool)
                mask[global_indices] = True
                segments_data.append([usd_path, mask])
                prim_covered |= mask

            prim_mask = torch.zeros(n_total, dtype=torch.bool)
            prim_mask[offset:offset + prim_size] = True
            remainder = prim_mask & ~prim_covered
            if remainder.any():
                segments_data.append([prim_path, remainder])
        else:
            mask = torch.zeros(n_total, dtype=torch.bool)
            mask[offset:offset + prim_size] = True
            segments_data.append([prim_path, mask])

    # Pass 2: strip common prefix from all raw names
    raw_names = [entry[0] for entry in segments_data]
    stripped = _strip_common_usd_prefix(raw_names)

    # Pass 3: build DisjointSegmentation
    # Skip any segment named 'background': DisjointSegmentation manages it automatically
    # as the complement of all named segments.
    segmentation = DisjointSegmentation(n_total, device)
    for name, (_, mask) in zip(stripped, segments_data):
        if name == DisjointSegmentation.BACKGROUND_NAME:
            continue
        segmentation.add_segment(name, mask.to(device))

    return GaussianSplatInput(merged), segmentation
