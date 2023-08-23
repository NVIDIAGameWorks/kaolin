# Copyright (c) 2019,20-21-23 NVIDIA CORPORATION & AFFILIATES.
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

import os
import warnings
from collections import namedtuple
import numpy as np
from tqdm import tqdm

import torch

try:
    from pxr import Usd, UsdGeom, Vt, Sdf, UsdShade
except ImportError:
    pass

from kaolin.io import materials as usd_materials
from kaolin.io import utils
from kaolin.rep import SurfaceMesh

from .utils import _get_stage_from_maybe_file, get_scene_paths, create_stage


__all__ = [
    'import_mesh',
    'import_meshes',
    'add_mesh',
    'export_mesh',
    'export_meshes',
    'get_raw_mesh_prim_geometry',
    'get_mesh_prim_materials',
]


def get_uvmap_primvar(mesh_prim):
    """Get UV coordinates primvar.

    In this order:
        - Look for primvar `st`, return if defined
        - Look for primvar with type TexCoord2fArray, return first found
        - Look for primvar named UVMap (seems to be true of blender exports)
        - Look for primvar with type Float2Array, return first found
    """
    primvars = UsdGeom.PrimvarsAPI(mesh_prim)
    mesh_st = primvars.GetPrimvar('st')
    if mesh_st.IsDefined():
        return mesh_st

    vec2f_primvars = []
    for pv in primvars.GetPrimvars():
        if pv.GetTypeName() == Sdf.ValueTypeNames.TexCoord2fArray:
            mesh_st = pv
            break
        elif pv.GetTypeName() == Sdf.ValueTypeNames.Float2Array:
            vec2f_primvars.append(pv)

    # This seems to be true for blender exports
    if not mesh_st.IsDefined():
        mesh_st = primvars.GetPrimvar('UVMap')

    if not mesh_st.IsDefined():
        if len(vec2f_primvars) > 1:
            mesh_st = vec2f_primvars[0]

    return mesh_st


def get_raw_mesh_prim_geometry(mesh_prim, time=None, with_normals=False, with_uvs=False):
    """ Extracts raw geometry properties from a mesh prim, converting them to torch tensors.

    Args:
        mesh_prim: USD Prim that should be of type Mesh
        time: timecode to extract values for
        with_normals (bool): if set, will extract normals (default: False)
        with_uvs (bool): if set, will also extract UV information (default: False)
        time (convertible to float, optional): positive integer indicating the time at which to retrieve parameters

    Returns:

    (dict):

        - **vertices** (torch.FloatTensor): vertex positions with any transforms already applied, of shape (N, 3)
        - **transform** (torch.FloatTensor): applied transform of shape (4, 4)
        - **faces** (torch.LongTensor): face vertex indices of original shape saved in the USD
        - **face_sizes** (torch.LongTensor): face vertex counts of original shape saved in the USD
        - **normals** (torch.FloatTensor, optional):
            if `with_normals=True`, normal values of original shape saved in the USD
        - **normals_interpolation** (string, optional):
            if `with_normals=True`, normal interpolation type saved in the USD, such as "faceVarying"
        - **uvs** (torch.FloatTensor, optional): if ``with_uvs=True``, raw UV values saved in the USD
        - **face_uvs_idx** (torch.LongTensor, optional):
            if ``with_uvs=True``, raw indices into the UV for every vertex of every face
        - **uv_interpolation** (string, optional):
            if ``with_uvs=True``, UV interpolation type saved in the USD, such as "faceVarying"
    """
    if time is None:
        time = Usd.TimeCode.Default()

    mesh = UsdGeom.Mesh(mesh_prim)

    # Vertices
    vertices = mesh.GetPointsAttr().Get(time=time)
    transform = torch.from_numpy(
        np.array(UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(time), dtype=np.float32))

    def _apply_transform(in_tensor):
        """ Applies local-to-world transform to in_tensor of shape N x 3. """
        tensor_homo = torch.nn.functional.pad(in_tensor, (0, 1), mode='constant', value=1.)
        return (tensor_homo @ transform)[:, :3]

    if vertices:
        vertices = torch.from_numpy(np.array(vertices, dtype=np.float32))
        vertices = _apply_transform(vertices)
    else:
        vertices = torch.zeros((0, 3), dtype=torch.float32)

    # Faces
    face_sizes = mesh.GetFaceVertexCountsAttr().Get(time=time)
    if face_sizes:
        face_sizes = torch.from_numpy(np.array(face_sizes, dtype=np.int64))
    else:
        face_sizes = torch.zeros((0,), dtype=torch.int64)
    faces = mesh.GetFaceVertexIndicesAttr().Get(time=time)
    if faces:
        faces = torch.from_numpy(np.array(faces, dtype=np.int64))
    else:
        faces = torch.zeros((0,), dtype=torch.int64)

    # Normals
    normals = None
    normals_interpolation = None
    if with_normals:
        normals = mesh.GetNormalsAttr().Get(time=time) or None
        normals_interpolation = mesh.GetNormalsInterpolation()
        if normals:
            normals = torch.from_numpy(np.array(normals, dtype=np.float32))
            normals = _apply_transform(normals)

    # UVs
    uvs = None
    uv_idx = None
    uv_interpolation = None
    if with_uvs:
        mesh_st = get_uvmap_primvar(mesh_prim)
        if mesh_st:
            uvs = mesh_st.Get(time=time) or None
            uv_idx = mesh_st.GetIndices(time=time) or None  # For faces or for vertices
            uv_interpolation = mesh_st.GetInterpolation()
            if uvs is not None:
                uvs = torch.from_numpy(np.array(uvs, dtype=np.float32))
            if uv_idx is not None:  # Note fails to convert empty array if just checking `if uv_idx`
                uv_idx = torch.from_numpy(np.array(uv_idx, dtype=np.int64))

    result = {'vertices': vertices,
              'transform': transform,
              'faces': faces,
              'face_sizes': face_sizes}
    if with_normals:
        result['normals'] = normals
        result['normals_interpolation'] = normals_interpolation
    if with_uvs:
        result['uvs'] = uvs
        result['uv_idx'] = uv_idx
        result['uv_interpolation'] = uv_interpolation
    return result


def get_mesh_prim_materials(mesh_prim, stage_dir, num_faces, time=None):
    """ Extracts and parses materials for a mesh_prim; currently only works for prims with a
    corresponding stage path (needs to be addressed).

    Args:
        mesh_prim: USD Prim that should be of type Mesh
        stage_dir: root stage directory
        num_faces: number of faces
        time: timecode to extract values for

    Returns:
    (tuple) containing two items:

        - **materials_dict** (dict from string to material): mapping from material name to material
        - **material_assignments_dict** (dict of str to torch.LongTensor): mapping from material name
            to face indices assigned to that material
    """
    if time is None:
        time = Usd.TimeCode.Default()

    # TODO: what is this ref path and is it set in the right way?
    metadata = mesh_prim.GetMetadata('references')
    ref_path = ''
    if metadata:
        ref_path = os.path.dirname(metadata.GetAddedOrExplicitItems()[0].assetPath)

    mesh_subsets = UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(mesh_prim)) or []
    mesh_material = UsdShade.MaterialBindingAPI.Apply(mesh_prim).ComputeBoundMaterial()[0]

    # Parse mesh materials
    materials = {}
    assignments = {}

    def _read_material_catch_errors(mat, mat_path):
        mesh_material_path = str(mat.GetPath())
        if mesh_material_path in materials:
            return mesh_material_path, materials[mesh_material_path]

        res = None
        try:
            # TODO: can we also read it from a loaded USD, no path?
            res = usd_materials.MaterialManager.read_usd_material(mat, mat_path, time)
            materials[mesh_material_path] = res
        except usd_materials.MaterialNotSupportedError as e:
            warnings.warn(e.args[0])
        except usd_materials.MaterialLoadError as e:
            warnings.warn(e.args[0])

        return mesh_material_path, res

    if mesh_material:
        name, _ = _read_material_catch_errors(mesh_material, stage_dir)
        assignments[name] = torch.arange(0, num_faces).to(torch.int64)

    for subset in mesh_subsets:
        subset_material, _ = UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).ComputeBoundMaterial()
        subset_material_metadata = subset_material.GetPrim().GetMetadata('references')
        mat_ref_path = ref_path
        if subset_material_metadata:
            asset_path = subset_material_metadata.GetAddedOrExplicitItems()[0].assetPath
            mat_ref_path = os.path.join(ref_path, os.path.dirname(asset_path))
        if not os.path.isabs(mat_ref_path):
            mat_ref_path = os.path.join(stage_dir, mat_ref_path)

        name, _ = _read_material_catch_errors(subset_material, mat_ref_path)
        # Note: down the line might want to check that subset.GetElementTypeAttr() == 'face', but it is currently
        # the only option supported by USD standard.
        assignments[name] = torch.from_numpy(np.array(subset.GetIndicesAttr().Get())).to(torch.int64)

    return materials, assignments


def get_face_uvs_idx(faces, face_sizes, uvs, uv_idx, uv_interpolation, **kwargs):
    if uv_interpolation in ['vertex', 'varying']:
        if uv_idx is None:
            # for vertex and varying interpolation, length of uv_idx should match
            # length of mesh_vertex_indices
            if uvs is not None:
                uv_idx = torch.tensor(list(range(len(uvs))))
            else:
                raise ValueError('Neither uvs nor uv_idx are set')
        face_uvs_idx = uv_idx[faces]
    elif uv_interpolation == 'faceVarying':
        if uv_idx is None:
            # for faceVarying interpolation, length of uv_idx should match num_faces * face_size
            uv_idx = torch.tensor([i for i in range(sum(face_sizes))])
        face_uvs_idx = uv_idx
    # TODO: implement uniform interpolation uv_interpolation == 'uniform':
    else:
        raise NotImplementedError(f'Interpolation type {uv_interpolation} is '
                                  'not supported')
    return face_uvs_idx


def get_face_normals(normals, normals_interpolation, **kwargs):
    if normals_interpolation == 'faceVarying':
        return normals
    else:
        raise NotImplementedError(f'Interpolation type {normals_interpolation} is '
                                  'not supported')


def _get_flattened_mesh_attributes(stage, scene_path, with_materials, with_normals, time):
    """Return attributes of all mesh prims under `scene_path`, flattened into a single mesh.

    Args:
        stage: USD stage
        scene_path (str): path from which to extract all child meshes
        with_materials (bool): if to parse materials
        with_normals (bool): if to parse normals
        time: time code

    Returns:

    (dict) containing at least:

         - **vertices** (torch.FloatTensor or None): vertex positions with any transforms already applied,
            of shape (N, 3) aggergated for all meshes under scene_path
        - **faces** (torch.LongTensor or None): face vertex indices of all meshes under scene_path
        - **face_sizes** (torch.LongTensor or None): face vertex counts of all meshes under scene_path
        - **face_normals** (torch.FloatTensor or None): normal values of all vertices of all faces of all meshes,
            set to `None` if `not with_normals` or normals are not available for some meshes under scene_path.
        - **uvs** (torch.FloatTensor or None): raw UV values of all meshes under scene_path, set to `None`
            if uvs are not available for some meshes.
        - **face_uvs_idx** (torch.LongTensor or None): processed indices into `uvs` for every vertex of
            every face.
        - **materials_dict** (None or dict from string to material): mapping from material name to material or
            None if not `with_materials`.
        - **material_assignments_dict** (None or dict of str to torch.LongTensor): mapping from material name
            to face indices assigned to that material or None if not `with_materials`

    may contain other auxiliary information, such as:
        - **normals_interpolation** (list of string or None): normal interpolation types for all meshes
        - **uv_idx** (list of torch.LongTensor or None): raw UV indices stored in the USD
        - **uv_interpolation** (list of string or None): UV interpolation types of all meshes
    """
    stage_dir = os.path.dirname(str(stage.GetRootLayer().realPath))
    prim = stage.GetPrimAtPath(scene_path)
    if not prim:
        raise ValueError(f'No prim found at "{scene_path}".')

    def _process_mesh_prim(mesh_prim, attrs, time):
        start_vertex_idx = sum([len(v) for v in attrs.get('vertices', [])])
        start_uv_idx = sum([len(u) for u in attrs.get('uvs', [])])
        start_face_idx = sum([len(f) for f in attrs.get('face_sizes', [])])

        # Returns dict of attributes:
        #   - vertices, transform, faces, face_sizes (never None)
        #   - normals, normals_interpolation, uvs, uv_idx, uv_interpolation  (sometimes None)
        geo = get_raw_mesh_prim_geometry(mesh_prim, time=time, with_normals=with_normals, with_uvs=True)
        if geo.get('uvs') is not None:
            geo['face_uvs_idx'] = get_face_uvs_idx(**geo) + start_uv_idx
        else:
            geo['face_uvs_idx'] = None
        geo['faces'] += start_vertex_idx

        if geo.get('normals') is not None:
            geo['face_normals'] = get_face_normals(**geo)
            del geo['normals']  # save memory

        for k, v in geo.items():
            attrs.setdefault(k, []).append(v)

        # Parse mesh materials
        if with_materials:
            num_faces = len(geo['face_sizes'])
            materials_dict, material_assignments_dict = get_mesh_prim_materials(
                mesh_prim, stage_dir, num_faces, time=time)
            attrs.setdefault('materials_dict', {}).update(materials_dict)
            attrs.setdefault('material_assignments_dict', {})
            for mat_name, face_ids in material_assignments_dict.items():
                face_ids = face_ids + start_face_idx
                if mat_name not in attrs['material_assignments_dict']:
                    attrs['material_assignments_dict'][mat_name] = face_ids
                else:
                    attrs['material_assignments_dict'][mat_name] = torch.cat(
                        [attrs['material_assignments_dict'][mat_name], face_ids])

    def _traverse(cur_prim, ref_path, attrs, time):
        if UsdGeom.Mesh(cur_prim):
            _process_mesh_prim(cur_prim, attrs, time)
        for child in cur_prim.GetChildren():
            _traverse(child, ref_path, attrs, time)

    attrs = {}
    _traverse(stage.GetPrimAtPath(scene_path), '', attrs, time)

    # Flatten obtained geometric attributes
    for k in ['vertices', 'faces', 'face_sizes', 'face_normals', 'uvs', 'face_uvs_idx']:
        value = attrs.get(k, [])
        if not all([v is not None for v in value]):  # Only applicable for normals and uvs
            warnings.warn(f'Some child prims for {scene_path} are missing {k}; skipping importing {k}.', UserWarning)
            attrs[k] = None
        elif len(value) == 0:
            attrs[k] = None
        else:
            attrs[k] = torch.cat(value)

    # Make sure these items are present as None
    for k in ['materials_dict', 'material_assignments_dict']:
        if k not in attrs:
            attrs[k] = None

    return attrs


def import_mesh(file_path_or_stage, scene_path=None, with_materials=False, with_normals=False,
                heterogeneous_mesh_handler=None, time=None, triangulate=False):
    r"""Import a single mesh from a USD file of Stage in an unbatched representation.

    Supports homogeneous meshes (meshes with consistent numbers of vertices per face).
    All sub-meshes found under the `scene_path` are flattened to a single mesh. The following
    interpolation types are supported for UV coordinates: `vertex`, `varying` and `faceVarying`.
    Returns an unbatched attributes as CPU torch tensors in an easy-to-manage
    :class:`kaolin.rep.SurfaceMesh` container.

    Args:
        file_path_or_stage (str, Usd.Stage):
            Path to usd file (`\*.usd`, `\*.usda`) or :class:`Usd.Stage`.
        scene_path (str, optional): Scene path within the USD file indicating which primitive to import.
            If not specified, the all meshes in the scene will be imported and flattened into a single mesh.
        with_materials (bool): if True, load materials. Default: False.
        with_normals (bool): if True, load vertex normals. Default: False.
        heterogeneous_mesh_handler (Callable, optional):
            function that handles a heterogeneous mesh, homogenizing, returning None or throwing error,
            with the following signature:
            ``heterogeneous_mesh_handler(vertices, face_vertex_counts, *args, face_assignments)``
            for example, see :func:`mesh_handler_naive_triangulate <kaolin.io.utils.mesh_handler_naive_triangulate>`
            and :func:`heterogeneous_mesh_handler_skip <kaolin.io.utils.heterogeneous_mesh_handler_skip>`.
            Default: will raise a NonHomogeneousMeshError.
        time (convertible to float, optional): Positive integer indicating the time at which to retrieve parameters.
        triangulate: if True, will triangulate all non-triangular meshes using same logic as
            :func:`mesh_handler_naive_triangulate <kaolin.io.utils.mesh_handler_naive_triangulate>`.

    Returns:
        (SurfaceMesh):
            an unbatched instance of :class:`kaolin.rep.SurfaceMesh`, where:

            * **normals** and **face_normals_idx** will only be filled if `with_normals=True`
            * **materials** will be a list
              of :class:`kaolin.io.materials.Material` sorted by their `material_name`;
              filled only if `with_materials=True`.
            * **material_assignments** will be a tensor
              of shape ``(num_faces,)`` containing the index
              of the material (in the `materials` list) assigned to the corresponding face,
              or `-1` if no material was assigned; filled only if `with_materials=True`.

    .. rubric:: Examples

    To load a mesh without loading normals or materials::

        >>> from kaolin.io.usd.mesh import import_mesh
        >>> mesh = import_mesh("sample_data/meshes/pizza.usda")
        >>> print(mesh)
        SurfaceMesh object with batching strategy NONE
                    vertices: [482, 3] (torch.float32)[cpu]
                         uvs: [2880, 2] (torch.float32)[cpu]
                       faces: [960, 3] (torch.int64)[cpu]
                face_uvs_idx: [960, 3] (torch.int64)[cpu]
               face_vertices: if possible, computed on access from: (faces, vertices)
                face_normals: if possible, computed on access from: (normals, face_normals_idx) or (vertices, faces)
              vertex_normals: if possible, computed on access from: (faces, face_normals)
                    face_uvs: if possible, computed on access from: (uvs, face_uvs_idx)

        >>> mesh.face_normals  # Causes face_normals and any attributes required to compute it to be auto-computed
        >>> mesh.to_batched()  # Apply fixed topology batching, unsqueezing most attributes
        >>> mesh = mesh.cuda(attributes=["vertices"])  # Moves just vertices to GPU
        >>> print(mesh)
        SurfaceMesh object with batching strategy FIXED
                    vertices: [1, 482, 3] (torch.float32)[cuda:0]
               face_vertices: [1, 960, 3, 3] (torch.float32)[cpu]
                face_normals: [1, 960, 3, 3] (torch.float32)[cpu]
                         uvs: [1, 2880, 2] (torch.float32)[cpu]
                       faces: [960, 3] (torch.int64)[cpu]
                face_uvs_idx: [1, 960, 3] (torch.int64)[cpu]
              vertex_normals: if possible, computed on access from: (faces, face_normals)
                    face_uvs: if possible, computed on access from: (uvs, face_uvs_idx)


    To load a mesh with normals and materials, while triangulating and homogenizing if needed::

        >>> from kaolin.io.usd.mesh import import_mesh
        >>> from kaolin.io.utils import mesh_handler_naive_triangulate
        >>> mesh = import_mesh("sample_data/meshes/pizza.usda",
                              with_normals=True, with_materials=True,
                              heterogeneous_mesh_handler=mesh_handler_naive_triangulate,
                              triangulate=True)
        >>> print(mesh)
        SurfaceMesh object with batching strategy NONE
                    vertices: [482, 3] (torch.float32)[cpu]
                face_normals: [960, 3, 3] (torch.float32)[cpu]
                         uvs: [2880, 2] (torch.float32)[cpu]
                       faces: [960, 3] (torch.int64)[cpu]
                face_uvs_idx: [960, 3] (torch.int64)[cpu]
        material_assignments: [960] (torch.int16)[cpu]
                   materials: list of length 2
               face_vertices: if possible, computed on access from: (faces, vertices)
              vertex_normals: if possible, computed on access from: (faces, face_normals)
                    face_uvs: if possible, computed on access from: (uvs, face_uvs_idx)
    """
    # TODO  add arguments to selectively import UVs
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    if scene_path is None:
        scene_path = stage.GetPseudoRoot().GetPath()
    if time is None:
        time = Usd.TimeCode.Default()
    meshes_list = import_meshes(stage, [scene_path],
                                heterogeneous_mesh_handler=heterogeneous_mesh_handler,
                                with_materials=with_materials,
                                with_normals=with_normals, times=[time], triangulate=triangulate)
    return meshes_list[0]


def import_meshes(file_path_or_stage, scene_paths=None, with_materials=False, with_normals=False,
                  heterogeneous_mesh_handler=None, times=None, triangulate=False):
    r"""Import one or more meshes from a USD file or Stage in an unbatched representation.

    Supports homogeneous meshes (meshes with consistent numbers of vertices per face). Custom handling of
    heterogeneous meshes can be achieved by passing a function through the ``heterogeneous_mesh_handler`` argument.
    The following interpolation types are supported for UV coordinates: `vertex`, `varying` and `faceVarying`.
    For each scene path specified in `scene_paths`, sub-meshes (if any) are flattened to a single mesh.
    Prims with no meshes or with heterogenous faces are skipped. Returns an unbatched attributes as CPU torch
    tensors in a list of easy-to-manage :class:`kaolin.rep.SurfaceMesh` containers.

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (`\*.usd`, `\*.usda`) or :class:`Usd.Stage`.
        scene_paths (list of str, optional): Scene path(s) within the USD file indicating which primitive(s)
            to import. If None, all prims of type `Mesh` will be imported.
        with_materials (bool): if True, load materials. Default: False.
        with_normals (bool): if True, load vertex normals. Default: False.
        heterogeneous_mesh_handler (Callable, optional):
            function that handles a heterogeneous mesh, homogenizing, returning None or throwing error,
            with the following signature:
            ``heterogeneous_mesh_handler(vertices, face_vertex_counts, *args, face_assignments)``
            for example, see :func:`mesh_handler_naive_triangulate <kaolin.io.utils.mesh_handler_naive_triangulate>`
            and :func:`heterogeneous_mesh_handler_skip <kaolin.io.utils.heterogeneous_mesh_handler_skip>`.
            Default: will raise a NonHomogeneousMeshError.
        times (list of int): Positive integers indicating the time at which to retrieve parameters.
        triangulate: if True, will triangulate all non-triangular meshes using same logic as
            :func:`mesh_handler_naive_triangulate <kaolin.io.utils.mesh_handler_naive_triangulate>`.

    Returns:
        (a list of SurfaceMesh):
            a list of unbatched instances of :class:`kaolin.rep.SurfaceMesh`, where:

            * **normals** and **face_normals_idx** will only be filled if `with_normals=True`
            * **materials** will be a list
              of :class:`kaolin.io.materials.Material` sorted by their `material_name`;
              filled only if `with_materials=True`.
            * **material_assignments** will be a tensor
              of shape ``(num_faces,)`` containing the index
              of the material (in the `materials` list) assigned to the corresponding face,
              or `-1` if no material was assigned; filled only if `with_materials=True`.

    .. rubric:: Examples

    To export and then import USD meshes::

        >>> # Create a stage with some meshes
        >>> vertices_list = [torch.rand(3, 3) for _ in range(3)]
        >>> faces_list = [torch.tensor([[0, 1, 2]]) for _ in range(3)]
        >>> stage = export_meshes('./new_stage.usd', vertices=vertices_list, faces=faces_list)
        >>> # Import meshes
        >>> meshes = import_meshes('./new_stage.usd')
        >>> len(meshes)
        3
        >>> meshes[0].vertices.shape
        torch.Size([3, 3])
        >>> [m.faces for m in meshes]
        [tensor([[0, 1, 2]]), tensor([[0, 1, 2]]), tensor([[0, 1, 2]])]

    To load multiple meshes from file, including materials and normals, while homongenizing and triangulating::

        >>> from kaolin.io.usd.mesh import import_meshes
        >>> from kaolin.io.utils import mesh_handler_naive_triangulate
        >>> meshes = import_meshes('sample_data/meshes/amsterdam.usda',
                                   with_normals=True, with_materials=True,
                                   heterogeneous_mesh_handler=mesh_handler_naive_triangulate,
                                   triangulate=True)
        >>> len(meshes)
        18
        >>> print(meshes[0])
        SurfaceMesh object with batching strategy NONE
                    vertices: [4, 3] (torch.float32)[cpu]
                face_normals: [2, 3, 3] (torch.float32)[cpu]
                         uvs: [4, 2] (torch.float32)[cpu]
                       faces: [2, 3] (torch.int64)[cpu]
                face_uvs_idx: [2, 3] (torch.int64)[cpu]
        material_assignments: [2] (torch.int16)[cpu]
                   materials: list of length 1
               face_vertices: if possible, computed on access from: (faces, vertices)
              vertex_normals: if possible, computed on access from: (faces, face_normals)
                    face_uvs: if possible, computed on access from: (uvs, face_uvs_idx)
        >>> # If needed, concatenate meshes into a batch
        >>> from kaolin.rep import SurfaceMesh
        >>> mesh = SurfaceMesh.cat(meshes, fixed_topology=False)
        >>> print(mesh)
        SurfaceMesh object with batching strategy LIST
            vertices: [
                      0: [4, 3] (torch.float32)[cpu]
                      1: [98, 3] (torch.float32)[cpu]
                      ...
                     17: [4, 3] (torch.float32)[cpu]
                      ]
        face_normals: [
                      0: [2, 3, 3] (torch.float32)[cpu]
                      ...
    """
    triangulate_handler = None if not triangulate else utils.mesh_handler_naive_triangulate

    # TODO  add arguments to selectively import UVs and normals
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    # Remove `instanceable` flags
    # USD Scene Instances are an optimization to avoid duplicating mesh data in memory
    # Removing the instanceable flag allows for easy retrieval of mesh data
    for p in stage.Traverse():
        p.SetInstanceable(False)
    if scene_paths is None:
        scene_paths = get_scene_paths(stage, prim_types=['Mesh'])
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)

    results = []
    silence_tqdm = len(scene_paths) < 10  # Silence tqdm if fewer than 10 paths are found
    for scene_path, time in zip(tqdm(scene_paths, desc="Importing from USD", unit="mesh", disable=silence_tqdm), times):
        # Returns (any may be None):
        # vertices, faces, face_sizes, face_normals, uvs, face_uvs_idx, materials_dict, material_assignments_dict
        mesh_attr = _get_flattened_mesh_attributes(stage, scene_path, with_materials, with_normals, time=time)
        vertices = mesh_attr['vertices']
        faces = mesh_attr['faces']
        face_sizes = mesh_attr['face_sizes']
        face_normals = mesh_attr['face_normals']
        uvs = mesh_attr['uvs']
        face_uvs_idx = mesh_attr['face_uvs_idx']
        materials_dict = mesh_attr['materials_dict'] or {}
        material_assignments_dict = mesh_attr['material_assignments_dict'] or {}

        # Handle attributes that require faces
        nfaces = 0
        facesize = 0
        if faces is not None:
            if face_sizes is not None and face_sizes.shape[0] > 0:
                facesize = face_sizes[0]

            if not torch.all(face_sizes == facesize):
                if heterogeneous_mesh_handler is None:
                    raise utils.NonHomogeneousMeshError(
                        f'Mesh at {scene_path} is non-homogeneous '
                        f'and cannot be imported from {repr(file_path_or_stage)}.')
                else:
                    mesh = heterogeneous_mesh_handler(vertices, face_sizes, faces, face_uvs_idx, face_normals,
                                                      face_assignments=material_assignments_dict)
                    if mesh is None:
                        continue
                    vertices, face_sizes, faces, face_uvs_idx, face_normals, material_assignments_dict = mesh
                    facesize = faces.shape[-1]

            if triangulate_handler is not None and not torch.all(face_sizes == 3):
                mesh = triangulate_handler(vertices, face_sizes, faces, face_uvs_idx, face_normals,
                                           face_assignments=material_assignments_dict)
                if mesh is None:
                    continue
                vertices, face_sizes, faces, face_uvs_idx, face_normals, material_assignments_dict = mesh
                facesize = 3

            faces = faces.view(-1 if len(faces) > 0 else 0, facesize)  # Nfaces x facesize
            nfaces = faces.shape[0]

        # Process face-related attributes, correctly handling absence of face information
        if face_uvs_idx is not None and face_uvs_idx.size(0) > 0:
            uvs = uvs.reshape(-1, 2)
            face_uvs_idx = face_uvs_idx.reshape(-1, max(1, facesize))
        else:
            uvs = None
            face_uvs_idx = None

        if face_normals is not None and face_normals.size(0) > 0:
            face_normals = face_normals.reshape((nfaces, -1, 3) if nfaces > 0 else (-1, 1, 3))
        else:
            face_normals = None

        materials = None
        material_assignments = None
        if with_materials and nfaces > 0:
            # TODO: add support for custom material error_handler
            def _default_error_handler(error, **kwargs):
                raise error
            materials, material_assignments = usd_materials.process_materials_and_assignments(
                materials_dict, material_assignments_dict, _default_error_handler, nfaces,
                error_context_str=scene_path)

        results.append(SurfaceMesh(
            vertices=vertices, faces=faces, uvs=uvs, face_uvs_idx=face_uvs_idx, face_normals=face_normals,
            material_assignments=material_assignments, materials=materials,
            unset_attributes_return_none=True))  # for greater backward compatibility

    return results


def add_mesh(stage, scene_path, vertices=None, faces=None, uvs=None, face_uvs_idx=None, face_normals=None,
             material_assignments=None, materials=None, time=None):
    r"""Add a mesh to an existing USD stage. The stage is modified but not saved to disk.

    Args:
        stage (Usd.Stage): Stage onto which to add the mesh.
        scene_path (str): Absolute path of mesh within the USD file scene. Must be a valid ``Sdf.Path``.
        vertices (torch.FloatTensor, optional): Vertices with shape ``(num_vertices, 3)``.
        faces (torch.LongTensor, optional): Vertex indices for each face with shape ``(num_faces, face_size)``.
            Mesh must be homogenous (consistent number of vertices per face).
        uvs (torch.FloatTensor, optional): of shape ``(num_uvs, 2)``.
        face_uvs_idx (torch.LongTensor, optional): of shape ``(num_faces, face_size)``. If provided, ``uvs`` must also
            be specified.
        face_normals (torch.Tensor, optional): of shape ``(num_faces, face_size, 3)``.
        materials (list of Material): list of material objects
        material_assignments (torch.ShortTensor): of shape ``(num_faces,)`` containing index of the
            material (in the above list) assigned to the corresponding face, or `-1` if no material was assigned
        time (convertible to float, optional): Positive integer defining the time at which the supplied parameters
            correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> vertices = torch.rand(3, 3)
        >>> faces = torch.tensor([[0, 1, 2]])
        >>> stage = create_stage('./new_stage.usd')
        >>> mesh = add_mesh(stage, '/World/mesh', vertices, faces)
        >>> stage.Save()
    """
    if time is None:
        time = Usd.TimeCode.Default()

    usd_mesh = UsdGeom.Mesh.Define(stage, scene_path)

    if faces is not None:
        num_faces = faces.size(0)
        face_vertex_counts = [faces.size(1)] * num_faces
        faces_list = faces.view(-1).cpu().long().numpy()
        usd_mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts, time=time)
        usd_mesh.GetFaceVertexIndicesAttr().Set(faces_list, time=time)
    if vertices is not None:
        vertices_list = vertices.detach().cpu().float().numpy()
        usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices_list), time=time)
    if uvs is not None:
        uvs_list = uvs.view(-1, 2).detach().cpu().float().numpy()
        pv = UsdGeom.PrimvarsAPI(usd_mesh.GetPrim()).CreatePrimvar(
            'st', Sdf.ValueTypeNames.Float2Array)
        pv.Set(uvs_list, time=time)

        if vertices is not None and uvs.size(0) == vertices.size(0):
            pv.SetInterpolation('vertex')
        elif faces is not None and uvs.view(-1, 2).size(0) == faces.size(0):
            pv.SetInterpolation('uniform')
        else:
            pv.SetInterpolation('faceVarying')

    if face_uvs_idx is not None:
        if uvs is not None:
            pv.SetIndices(Vt.IntArray.FromNumpy(face_uvs_idx.view(-1).cpu().long().numpy()), time=time)
        else:
            warnings.warn('If providing "face_uvs_idx", "uvs" must also be provided.')

    if face_normals is not None:
        # Note: normals are stored as (num_faces * face_sizes) x 3 array
        face_normals = face_normals.view(-1, 3).cpu().float().numpy()
        usd_mesh.GetNormalsAttr().Set(face_normals, time=time)
        UsdGeom.PointBased(usd_mesh).SetNormalsInterpolation('faceVarying')

    if faces is not None and material_assignments is not None and materials is not None:
        stage.DefinePrim(f'{scene_path}/Looks', 'Scope')

        # Create submeshes
        for i, material in enumerate(materials):
            # Note: without int(x) for ... fails in Set with type mismatch
            face_idx = [int(x) for x in list((material_assignments == i).nonzero().squeeze().numpy())]
            subset_prim = stage.DefinePrim(f'{scene_path}/subset_{i}', 'GeomSubset')
            subset_prim.GetAttribute('indices').Set(face_idx)
            if isinstance(material, usd_materials.Material):
                # TODO: should be write_to_usd
                material._write_usd_preview_surface(stage, f'{scene_path}/Looks/material_{i}',
                                                    [subset_prim], time, texture_dir=f'material_{i}',
                                                    texture_file_prefix='')    # TODO allow users to pass root path to save textures to
    return usd_mesh.GetPrim()


def export_mesh(file_path, scene_path='/World/Meshes/mesh_0', vertices=None, faces=None,
                uvs=None, face_uvs_idx=None, face_normals=None, material_assignments=None, materials=None,
                up_axis='Y', time=None):
    r"""Export a single mesh to USD and save the stage to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_path (str, optional):
            Absolute path of mesh within the USD file scene. Must be a valid ``Sdf.Path``.
            If no path is provided, a default path is used.
        vertices (torch.FloatTensor, optional): Vertices with shape ``(num_vertices, 3)``.
        faces (torch.LongTensor, optional):
            Vertex indices for each face with shape ``(num_faces, face_size)``.
            Mesh must be homogenous (consistent number of vertices per face).
        uvs (torch.FloatTensor, optional): of shape ``(num_uvs, 2)``.
        face_uvs_idx (torch.LongTensor, optional):
            of shape ``(num_faces, face_size)``.
            If provided, `uvs` must also be specified.
        face_normals (torch.Tensor, optional): of shape ``(num_vertices, num_faces, 3)``.
        materials (list of Material): list of material objects
        material_assignments (torch.ShortTensor): of shape ``(num_faces,)`` containing index of the
            material (in the above list) assigned to the corresponding face, or `-1` if no material was assigned
        up_axis (str, optional): Specifies the scene's up axis. Choose from ``['Y', 'Z']``
        time (convertible to float, optional):
            Positive integer defining the time at which the supplied parameters correspond to.

    Returns:
       (Usd.Stage)

    Example:
        >>> vertices = torch.rand(3, 3)
        >>> faces = torch.tensor([[0, 1, 2]])
        >>> stage = export_mesh('./new_stage.usd', vertices=vertices, faces=faces)
    """
    assert isinstance(scene_path, str)
    if time is None:
        time = Usd.TimeCode.Default()
    if os.path.exists(file_path):
        stage = Usd.Stage.Open(file_path)
        UsdGeom.SetStageUpAxis(stage, up_axis)
    else:
        stage = create_stage(file_path, up_axis)
    add_mesh(stage, scene_path, vertices, faces, uvs, face_uvs_idx,
             face_normals, material_assignments, materials, time=time)
    stage.Save()

    return stage


def export_meshes(file_path, scene_paths=None, vertices=None, faces=None,
                  uvs=None, face_uvs_idx=None, face_normals=None, material_assignments=None, materials=None,
                  up_axis='Y', times=None):
    r"""Export multiple meshes to a new USD stage.

    Export multiple meshes defined by lists vertices and faces and save the stage to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_paths (list of str, optional): Absolute paths of meshes within the USD file scene. Must have the same
            number ofpaths as the number of meshes ``N``. Must be a valid Sdf.Path. If no path is provided, a default
            path is used.
        vertices (list of torch.FloatTensor, optional): Vertices with shape ``(num_vertices, 3)``.
        faces (list of torch.LongTensor, optional): Vertex indices for each face with shape ``(num_faces, face_size)``.
            Mesh must be homogenous (consistent number of vertices per face).
        uvs (list of torch.FloatTensor, optional): of shape ``(num_uvs, 2)``.
        face_uvs_idx (list of torch.LongTensor, optional): of shape ``(num_faces, face_size)``. If provided, `uvs`
            must also be specified.
        face_normals (list of torch.Tensor, optional): of shape ``(num_faces, face_size, 3)``.
        materials (list of list of Material): list of material objects
        material_assignments (list of torch.ShortTensor): of shape `(\text{num_faces},)` containing index of the
            material (in the above list) assigned to the corresponding face, or `-1` if no material was assigned
        up_axis (str, optional): Specifies the scene's up axis. Choose from ``['Y', 'Z']``.
        times (list of int, optional): Positive integers defining the time at which the supplied parameters
            correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> vertices_list = [torch.rand(3, 3) for _ in range(3)]
        >>> faces_list = [torch.tensor([[0, 1, 2]]) for _ in range(3)]
        >>> stage = export_meshes('./new_stage.usd', vertices=vertices_list, faces=faces_list)
    """
    stage = create_stage(file_path, up_axis)
    num_meshes = -1
    # TODO: might want to consider sharing materials
    for param in [scene_paths, vertices, faces, uvs, face_uvs_idx,
                  face_normals, material_assignments, materials, times]:
        if param is not None:
            if not type(param) == list:
                raise TypeError(f'Unexpected type {type(param)} input to export_meshes (list expected)')
            if num_meshes == -1:
                num_meshes = len(param)
            else:
                assert len(param) == num_meshes, f'All list inputs to export_meshes must have same length'

    if scene_paths is None:
        if not stage.GetPrimAtPath('/World/Meshes'):
            stage.DefinePrim('/World/Meshes', 'Xform')
        scene_paths = [f'/World/Meshes/mesh_{i}' for i in range(len(vertices))]

    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)

    for i, scene_path in enumerate(tqdm(scene_paths, desc="Exporting to USD", unit="mesh")):
        # Note: we make parameters explicit to ensure tests catch any API changes reliably
        add_mesh(stage, scene_path,
                 vertices=None if vertices is None else vertices[i],
                 faces=None if faces is None else faces[i],
                 uvs=None if uvs is None else uvs[i],
                 face_uvs_idx=None if face_uvs_idx is None else face_uvs_idx[i],
                 face_normals=None if face_normals is None else face_normals[i],
                 material_assignments=None if material_assignments is None else material_assignments[i],
                 materials=None if materials is None else materials[i],
                 time=times[i])
    stage.Save()

    return stage
