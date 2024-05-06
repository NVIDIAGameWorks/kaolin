# Copyright (c) 2022-24 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import os
from pathlib import Path
import posixpath
from PIL import Image
import torch
import warnings

__all__ = ["heterogeneous_mesh_handler_skip",
           "heterogeneous_mesh_handler_naive_homogenize",
           "mesh_handler_naive_triangulate",
           "NonHomogeneousMeshError",
           "TextureExporter",
           "write_image",
           "read_image"]


class NonHomogeneousMeshError(Exception):
    """Raised when expecting a homogeneous mesh but a heterogenous
    mesh is encountered.
    """

    __slots__ = ['message']

    def __init__(self, message):
        self.message = message


# Mesh Functions
def heterogeneous_mesh_handler_skip(*args, **kwargs):
    r"""Skip heterogeneous meshes."""
    return None


def heterogeneous_mesh_handler_naive_homogenize(*args, **kwargs):
    r"""Same as :func:`mesh_handler_naive_triangulate`, see docs.
    .. deprecated:: 0.14.0
    """
    warnings.warn("heterogeneous_mesh_handler_naive_homogenize is deprecated, "
                  "please use kaolin.io.utils.mesh_handler_naive_triangulate instead",
                  DeprecationWarning, stacklevel=2)
    return mesh_handler_naive_triangulate(*args, **kwargs)


def mesh_handler_naive_triangulate(vertices, face_vertex_counts, *features, face_assignments=None):
    r"""Triangulate a list of faces containing polygons of varying number of edges using naive fan
    triangulation.

    Args:
        vertices (torch.FloatTensor): Vertices with shape ``(N, 3)``.
        face_vertex_counts (torch.LongTensor): Number of vertices for each face with shape ``(M)``
            for ``M`` faces.
        features: Variable length features that need to be handled as 1D Tensor ``(num_face_vertices)``,
            with one feature per face vertex. For example, faces as a tensor
            ``[face0_vertex0_id, face0_vertex1_id, face0_vertex2_id, face1_vertex0_id...]`` or as UV indices:
            ``[face0_vertex0_uv_idx, face0_vertex1_uv_idx, ...]``.
        face_assignments (dict): mapping from key to torch.LongTensor, where each value of the tensor corresponds
            to a face index. These indices will be expanded and rewritten to include triangulated face indices.
            Two modes are supported for face_assignments:
            1) if 1D tensor, each face idx will be replaced with indices of faces it was split into
            2) if 2D tensor, expects shape (K, 2), where [x, i] will be replaced with index of the first face
            [x, i] was split into, effectively supporting tensors containing (start,end].
    Returns:
        (tuple):
            Homogeneous list of attributes with exactly same type and number as function inputs.

            - **vertices** (torch.Tensor): unchanged `vertices` of shape ``(N, 3)``
            - **face_vertex_counts** (torch.LongTensor): tensor of length ``new_num_faces`` filled with 3.
            - **features** (torch.Tensor): of same type as input and shape ``(new_num_faces, 3)``
            - **face_assignments** (dict): returned only if face_assignments is set, with each value containing
                    new face indices equivalent to the prior assignments (see two modes for ``face_assignments``)
    """
    def _homogenize(attr, face_vertex_counts):
        if attr is not None:
            attr = attr if isinstance(attr, list) else attr.tolist()
            idx = 0
            new_attr = []
            for face_vertex_count in face_vertex_counts:
                attr_face = attr[idx:(idx + face_vertex_count)]
                idx += face_vertex_count
                while len(attr_face) >= 3:
                    new_attr.append(attr_face[:3])
                    attr_face.pop(1)
            return torch.tensor(new_attr)
        else:
            return None

    def _homogenize_counts(face_vertex_counts, compute_face_id_mappings=False):
        mappings = []  # mappings[i] = [new face ids that i was split into]
        num_faces = 0
        for face_vertex_count in face_vertex_counts:
            attr_face = list(range(0, face_vertex_count))
            new_indices = []
            while len(attr_face) >= 3:
                if compute_face_id_mappings:
                    new_indices.append(num_faces)
                num_faces += 1
                attr_face.pop(1)
            if compute_face_id_mappings:
                mappings.append(new_indices)
        return torch.full((num_faces,), 3, dtype=torch.long), mappings

    new_attrs = [_homogenize(a, face_vertex_counts) for a in features]
    new_counts, face_idx_mappings = _homogenize_counts(face_vertex_counts,
                                                       face_assignments is not None and len(face_assignments) > 0)

    if face_assignments is None:
        # Note: for python > 3.8 can do "return vertices, new_counts, *new_attrs"
        return tuple([vertices, new_counts] + new_attrs)

    # TODO: this is inefficient and could be improved
    new_assignments = {}
    for k, v in face_assignments.items():
        if len(v.shape) == 1:
            new_idx = []
            for old_idx in v:
                new_idx.extend(face_idx_mappings[old_idx])
            new_idx = torch.LongTensor(new_idx)
        else:
            # We support this (start, end] mode for efficiency of OBJ readers
            assert len(v.shape) == 2 and v.shape[1] == 2, 'Expects shape (K,) or (K, 2) for face_assignments'
            new_idx = torch.zeros_like(v)
            for row in range(v.shape[0]):
                old_idx_start = v[row, 0]
                old_idx_end = v[row, 1] - 1
                new_idx[row, 0] = face_idx_mappings[old_idx_start][0]
                new_idx[row, 1] = face_idx_mappings[old_idx_end][-1] + 1
        new_assignments[k] = new_idx

    # Note: for python > 3.8 can do "return vertices, new_counts, *new_attrs, new_assignments"
    return tuple([vertices, new_counts] + new_attrs + [new_assignments])


# TODO: consider finding the fastest implementation, e.g. https://pytorch.org/vision/stable/io.html#images
def write_image(img_tensor, path):
    """ Writes PyTorch image tensor to file. Will create containing directory if does not exist.

    Args:
        img_tensor (torch.Tensor): tensor that is uint8 0..255 or float 0..1, either chw or hwc format,
            can be a batch of one, e.g. of shape `(1, 3, H, W)`
        path (str): image path where to save image; will overwrite existing

    Returns:

    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if torch.is_floating_point(img_tensor):
        img_tensor = (img_tensor * 255.).clamp_(0, 255).to(torch.uint8)
    if img_tensor.ndim > 3:
        img_tensor = img_tensor.squeeze(0)
    if img_tensor.ndim < 3:
        img_tensor = img_tensor.unsqueeze(-1)
    likely_chw = img_tensor.shape[0] in [1, 3, 4] and img_tensor.shape[-1] not in [1, 3, 4]
    if likely_chw:
        img_tensor = img_tensor.permute(1, 2, 0)
    img = Image.fromarray(img_tensor.squeeze().detach().cpu().numpy())
    img.save(path)


def read_image(path):
    """ Reads image from path. Note that this way is order of magnitude faster than some other ways;
    use this function rather than writing your own.

    Args:
        path (str): path where to read image from

    Returns:
        (torch.FloatTensor) in range 0..1 with shape `(height, width, num_channels)`
    """
    img = Image.open(str(path))
    # Note: > 10x faster than ((torch.FloatTensor(img.getdata())).reshape(*img.size, -1) / 255.).permute(2, 0, 1)
    res = torch.from_numpy(np.array(img))
    if len(res.shape) == 2:
        res = res.unsqueeze(-1)
    return res.float() / 255.


class TextureExporter:
    """ Utility functor that encapsulates logic about overwriting image files. Useful for e.g. saving textures
    without overwriting existing textures in a directory.

    Example:
        export_fn = TextureExporter(base_dir, 'textures', overwrite_textures=False)
        rel_path = export_fn(image1, 'image')
        print(rel_path)  # prints "textures/image.png"

        # Save another image with same basename
        rel_path = export_fn(image2, 'image')
        print(rel_path)  # prints "textures/image_0.png"

    Args:
            base_dir (str): base directory where to write images, must exist
            relative_dir (str, optional): if set, will create this subdirectory under base_dir and
                images will be saved there
            file_prefix (str, optional): prefix to add to filenames
            image_extension (str): extension to use for images; default is `.png`
            overwrite_files (bool): set to true to overwrite existing images; if False (default), will add an
                index to filename to ensure no image file is overwritten.
    """
    def __init__(self, base_dir, relative_dir='', file_prefix='', image_extension='.png',
                 overwrite_files=False):
        self.base_dir = base_dir
        self.relative_dir = Path(relative_dir).as_posix() if len(relative_dir) > 0 else ''
        self.file_prefix = file_prefix
        self.overwrite_files = overwrite_files
        self.image_extension = image_extension

    def _suggest_relative_filename(self, texture_file_basename, extension):
        idx = -1

        def _make_rel_path():
            idx_string = f'_{idx}' if idx >= 0 else ''
            filename = f'{self.file_prefix}{texture_file_basename}{idx_string}{extension}'
            return posixpath.join(self.relative_dir, filename)

        rel_filepath = _make_rel_path()
        if self.overwrite_files:
            return rel_filepath

        # Increment until we find a file that does not exist
        while os.path.exists(posixpath.join(self.base_dir, rel_filepath)):
            idx += 1
            rel_filepath = _make_rel_path()
        return rel_filepath

    def __call__(self, image, texture_file_basename, extension=None):
        """ Writes image, given its basename, constucting a path based on the functor options.
        The path relative to base_dir will be returned, taking the form of:
        `{relative_dir}/{file_prefix}{basename}{optional_index}{extension}`

        Args:
            image (torch.Tensor):  tensor that is uint8 0..255 or float 0..1, either chw or hwc format,
            can be a batch of one, e.g. of shape `(1, 3, H, W)`
            texture_file_basename (str): basename for the file, e.g. "diffuse"
            extension (str): pass in extension if a different one from default is wanted

        Returns:
            (str) relative path
        """
        if not os.path.exists(self.base_dir):
            raise ValueError(f'Base path does not exist {self.base_dir}')
        if len(self.relative_dir) > 0:
            os.makedirs(os.path.join(self.base_dir, self.relative_dir), exist_ok=True)

        if extension is None:
            extension = self.image_extension

        # TODO: why was posixpath used everywhere in USD export code? Test on Windows.
        rel_filepath = self._suggest_relative_filename(texture_file_basename, extension)
        write_image(image, posixpath.join(self.base_dir, rel_filepath))
        return rel_filepath
