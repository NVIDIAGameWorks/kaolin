# Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES.
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
import re
import warnings
import numpy as np
import pathlib
from tqdm import tqdm

import torch

from pxr import Usd, UsdGeom, Vt, Sdf, UsdShade

from kaolin.io.materials import MaterialLoadError, MaterialNotSupportedError, process_materials_and_assignments
from kaolin.io import utils
from kaolin.rep import SurfaceMesh

from .materials import export_material, UsdMaterialIoManager
from .subset import add_subset
from .utils import _get_stage_from_maybe_file, get_scene_paths, create_stage
from .transform import set_local_to_world_transform, get_local_to_world_transform


__all__ = [
    'get_mesh_scene_paths',
    'import_mesh',
    'import_meshes',
    'add_mesh',
    'export_mesh',
    'export_meshes',
    'get_raw_mesh_prim_geometry',
    'get_mesh_prim_materials',
]


def get_mesh_scene_paths(file_path_or_stage, scene_path=None):
    r"""Returns all mesh scene paths contained in specified file.

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        scene_path (str, optional): If specified, only return paths under this scene path prefix.

    Returns:
        (list of str): List of filtered scene paths.
    """
    scene_path_regex = f"{re.escape(scene_path)}(/|$)" if scene_path is not None else None
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        mesh_paths = get_scene_paths(stage, prim_types=['Mesh'], scene_path_regex=scene_path_regex)
    finally:
        del stage, file_path_or_stage
    return mesh_paths


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

        - **vertices** (torch.FloatTensor): vertex positions in the prim's local space, of shape (N, 3)
        - **transform** (torch.FloatTensor): local-to-world transform of shape (4, 4), in USD convention (translation in last row)
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


def get_mesh_prim_materials(mesh_prim, time=None):
    """ Extracts and parses materials for a mesh_prim; currently only works for prims with a
    corresponding stage path (needs to be addressed).

    Args:
        mesh_prim: USD Prim that should be of type Mesh
        time: timecode to extract values for

    Returns:
    (tuple) containing two items:

        - **materials_dict** (dict from string to material): mapping from material name to material
        - **material_assignments_dict** (dict of str to torch.LongTensor): mapping from material name
            to face indices assigned to that material
    """
    if time is None:
        time = Usd.TimeCode.Default()

    mesh_subsets = UsdGeom.Subset.GetGeomSubsets(
        UsdGeom.Imageable(mesh_prim), familyName='materialBind') or []
    mesh_material = UsdShade.MaterialBindingAPI.Apply(mesh_prim).ComputeBoundMaterial()[0]

    # Parse mesh materials
    materials = {}
    assignments = {}

    def _read_material_catch_errors(mat):
        mesh_material_path = str(mat.GetPath())
        if mesh_material_path in materials:
            return mesh_material_path, materials[mesh_material_path]

        res = None
        try:
            res = UsdMaterialIoManager.read_material(mat, time)
            materials[mesh_material_path] = res
        except MaterialNotSupportedError as e:
            warnings.warn(e.args[0])
        except MaterialLoadError as e:
            warnings.warn(e.args[0])

        return mesh_material_path, res

    if mesh_material:
        name, _ = _read_material_catch_errors(mesh_material)
        num_faces = UsdGeom.Mesh(mesh_prim).GetFaceCount(timeCode=time)
        assignments[name] = torch.arange(0, num_faces, dtype=torch.int64)

    for subset in mesh_subsets:
        subset_material, _ = UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).ComputeBoundMaterial()
        name, _ = _read_material_catch_errors(subset_material)
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


def set_normals(geo):
    normals_interpolation = geo.get('normals_interpolation')
    normals = geo.get('normals')
    if normals_interpolation == 'faceVarying':
        geo['face_normals'] = normals
    elif normals_interpolation == 'vertex':
        geo['vertex_normals'] = normals
    else:
        raise NotImplementedError(f'Interpolation type {normals_interpolation} is '
                                  'not supported')


def _get_mesh_prim_attributes(stage, scene_path, with_materials, with_normals, time):
    """Return attributes of the mesh prim at `scene_path`.

    Args:
        stage: USD stage
        scene_path (str): path to the mesh prim to read
        with_materials (bool): if to parse materials
        with_normals (bool): if to parse normals
        time: time code

    Returns:

    (dict) containing at least:

        - **vertices** (torch.FloatTensor or None): vertex positions in the prim's local space, of shape (N, 3)
        - **faces** (torch.LongTensor or None): face vertex indices
        - **face_sizes** (torch.LongTensor or None): face vertex counts
        - **face_normals** (torch.FloatTensor or None): normal values for all vertices of all faces,
            set to `None` if `not with_normals` or normals are not available.
        - **uvs** (torch.FloatTensor or None): raw UV values, set to `None` if uvs are not available.
        - **face_uvs_idx** (torch.LongTensor or None): processed indices into `uvs` for every vertex of every face.
        - **materials_dict** (None or dict from string to material): mapping from material name to material or
            None if not `with_materials`.
        - **material_assignments_dict** (None or dict of str to torch.LongTensor): mapping from material name
            to face indices assigned to that material or None if not `with_materials`
        - **transform** (torch.FloatTensor or None): local-to-world transform of shape (4, 4),
            or None if the prim has an identity transform.
    """
    stage_dir = os.path.dirname(str(stage.GetRootLayer().realPath))
    prim = stage.GetPrimAtPath(scene_path)
    if not prim:
        raise ValueError(f'No prim found at "{scene_path}".')

    geo = get_raw_mesh_prim_geometry(prim, time=time, with_normals=with_normals, with_uvs=True)
    attrs = {k: geo[k] for k in ('vertices', 'faces', 'face_sizes')}

    if geo.get('uvs') is not None:
        attrs['face_uvs_idx'] = get_face_uvs_idx(**geo)
        attrs['uvs'] = geo['uvs']
    else:
        attrs['face_uvs_idx'] = None
        attrs['uvs'] = None

    if geo.get('normals') is not None:
        set_normals(geo)
        attrs['face_normals'] = geo.get('face_normals')
        attrs['vertex_normals'] = geo.get('vertex_normals')
    else:
        attrs['face_normals'] = None
        attrs['vertex_normals'] = None

    if with_materials:
        num_faces = len(geo['face_sizes'])
        attrs['materials_dict'], attrs['material_assignments_dict'] = get_mesh_prim_materials(
            prim, time=time)
    else:
        attrs['materials_dict'] = None
        attrs['material_assignments_dict'] = None

    tfm = get_local_to_world_transform(stage, prim, time)
    attrs['transform'] = tfm.float() if tfm is not None else None

    return attrs


def import_mesh(file_path_or_stage, scene_path=None, with_materials=False, with_normals=False,
                heterogeneous_mesh_handler=None, time=None, triangulate=False):
    r"""Import a mesh scene from a USD file or Stage,
    all the prims under `scene_path` are merged into a single :class:`kaolin.rep.SurfaceMesh`
    with their local-to-world transforms applied.

    Supports homogeneous meshes (meshes with consistent numbers of vertices per face).
    The following interpolation types are supported for UV coordinates: `vertex`, `varying`
    and `faceVarying`.  Returns an unbatched :class:`kaolin.rep.SurfaceMesh` container.

    Args:
        file_path_or_stage (str, Usd.Stage):
            Path to usd file (`\*.usd`, `\*.usda`) or :class:`Usd.Stage`.
        scene_path (str, optional): If specified, only import meshes under this scene path prefix.
            Default: import all meshes in the file.
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
            If `heterogeneous_mesh_handler` is not set, this flag will cause non-homogeneous meshes to
            be triangulated and loaded without error; otherwise triangulation executes after `heterogeneous_mesh_handler`,
            which may skip or throw an error.

    Returns:
        (SurfaceMesh or None):
            A single merged unbatched :class:`kaolin.rep.SurfaceMesh` in world space,
            or ``None`` if no meshes are found.

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

    To load a mesh with normals and materials, while triangulating and homogenizing if needed::

        >>> from kaolin.io.usd.mesh import import_mesh
        >>> from kaolin.io.utils import mesh_handler_naive_triangulate
        >>> mesh = import_mesh("sample_data/meshes/pizza.usda",
                              with_normals=True, with_materials=True,
                              heterogeneous_mesh_handler=mesh_handler_naive_triangulate,
                              triangulate=True)
    """
    scene_paths = get_mesh_scene_paths(file_path_or_stage, scene_path=scene_path)
    meshes = import_meshes(file_path_or_stage, scene_paths,
                           with_materials=with_materials, with_normals=with_normals,
                           heterogeneous_mesh_handler=heterogeneous_mesh_handler,
                           times=None if time is None else [time] * len(scene_paths),
                           triangulate=triangulate, return_list=False)
    if not meshes:
        return None

    meshes_list = list(meshes.values())
    if len(meshes_list) == 1:
        return meshes_list[0].as_transformed()
    return SurfaceMesh.flatten(meshes_list, group_materials_by_name=True)


def import_meshes(file_path_or_stage, scene_paths=None, with_materials=False, with_normals=False,
                  heterogeneous_mesh_handler=None, times=None, triangulate=False, return_list=True):
    r"""Import one or more meshes from a USD file or Stage in an unbatched representation.

    Supports homogeneous meshes (meshes with consistent numbers of vertices per face). Custom handling of
    heterogeneous meshes can be achieved by passing a function through the ``heterogeneous_mesh_handler`` argument.
    The following interpolation types are supported for UV coordinates: `vertex`, `varying` and `faceVarying`.
    For each scene path specified in `scene_paths`, sub-meshes (if any) are flattened to a single mesh.

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (`\*.usd`, `\*.usda`) or :class:`Usd.Stage`.
        scene_paths (list of str, optional): Scene path(s) within the USD file indicating which primitive(s)
            to import. Default: all mesh prims in the file are imported.
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
            If `heterogeneous_mesh_handler` is not set, this flag will cause non-homogeneous meshes to
            be triangulated and loaded without error; otherwise triangulation executes after `heterogeneous_mesh_handler`,
            which may skip or throw an error.
        return_list (bool, optional): If ``True`` (default), return a ``list`` of
            :class:`kaolin.rep.SurfaceMesh` in the order of ``scene_paths`` (v0.18-compatible
            behaviour). If ``False``, return a ``dict`` keyed by scene path.

    Returns:
        (list or dict of SurfaceMesh):
            A list of :class:`kaolin.rep.SurfaceMesh` (when ``return_list=True``, the default)
            or a ``dict[str, SurfaceMesh]`` keyed by scene path (when ``return_list=False``),
            where each mesh has:

            * **normals** and **face_normals_idx** will only be filled if `with_normals=True`
            * **materials** will be a list
              of :class:`kaolin.io.materials.Material` sorted by their `material_name`;
              filled only if `with_materials=True`.
            * **material_assignments** will be a tensor
              of shape ``(num_faces,)`` containing the index
              of the material (in the `materials` list) assigned to the corresponding face,
              or `-1` if no material was assigned; filled only if `with_materials=True`.

            Meshes skipped due to ``heterogeneous_mesh_handler`` returning ``None`` are omitted.

    .. rubric:: Examples

    To export and then import USD meshes::

        >>> # Create a stage with some meshes
        >>> vertices_list = [torch.rand(3, 3) for _ in range(3)]
        >>> faces_list = [torch.tensor([[0, 1, 2]]) for _ in range(3)]
        >>> stage = export_meshes('./new_stage.usd', vertices=vertices_list, faces=faces_list)
        >>> # Import meshes (auto-discovers all mesh prims)
        >>> meshes = import_meshes('./new_stage.usd')
        >>> len(meshes)
        3
        >>> meshes[0].vertices.shape
        torch.Size([3, 3])

    To load multiple meshes from file, including materials and normals, while homongenizing and triangulating::

        >>> from kaolin.io.usd.mesh import import_meshes
        >>> from kaolin.io.utils import mesh_handler_naive_triangulate
        >>> meshes = import_meshes('sample_data/meshes/amsterdam.usda',
                                   with_normals=True, with_materials=True,
                                   heterogeneous_mesh_handler=mesh_handler_naive_triangulate,
                                   triangulate=True)
        >>> len(meshes)
        18
        >>> # If needed, concatenate meshes into a batch
        >>> from kaolin.rep import SurfaceMesh
        >>> mesh = SurfaceMesh.cat(meshes, fixed_topology=False)
    """
    triangulate_handler = None if not triangulate else utils.mesh_handler_naive_triangulate
    if heterogeneous_mesh_handler is None:
        heterogeneous_mesh_handler = triangulate_handler

    # TODO  add arguments to selectively import UVs and normals
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        if scene_paths is None:
            scene_paths = get_mesh_scene_paths(stage)
        # Remove `instanceable` flags
        # USD Scene Instances are an optimization to avoid duplicating mesh data in memory
        # Removing the instanceable flag allows for easy retrieval of mesh data
        # TODO(cfujitsang): get_scene_paths should support proxy, and material binding should also work with proxy. See commit 84853b46
        for p in stage.Traverse():
            p.SetInstanceable(False)
        if times is None:
            times = [Usd.TimeCode.Default()] * len(scene_paths)

        results = {}
        silence_tqdm = len(scene_paths) < 10  # Silence tqdm if fewer than 10 paths are found
        for scene_path, time in zip(tqdm(scene_paths, desc="Importing from USD", unit="mesh", disable=silence_tqdm), times):
            # Returns (any may be None):
            # vertices, faces, face_sizes, face_normals, uvs, face_uvs_idx, materials_dict, material_assignments_dict
            mesh_attr = _get_mesh_prim_attributes(stage, scene_path, with_materials, with_normals, time=time)
            vertices = mesh_attr['vertices']
            faces = mesh_attr['faces']
            face_sizes = mesh_attr['face_sizes']
            face_normals = mesh_attr['face_normals']
            vertex_normals = mesh_attr.get('vertex_normals')
            uvs = mesh_attr['uvs']
            face_uvs_idx = mesh_attr['face_uvs_idx']
            materials_dict = mesh_attr['materials_dict'] or {}
            material_assignments_dict = mesh_attr['material_assignments_dict'] or {}
            transform = mesh_attr.get('transform')

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
                uvs[..., 1] = 1 - uvs[..., 1]  # modify according to convention
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
                materials, material_assignments = process_materials_and_assignments(
                    materials_dict, material_assignments_dict, _default_error_handler, nfaces,
                    error_context_str=scene_path)

            results[str(scene_path)] = SurfaceMesh(
                vertices=vertices, faces=faces, uvs=uvs, face_uvs_idx=face_uvs_idx, face_normals=face_normals,
                vertex_normals=vertex_normals,
                material_assignments=material_assignments, materials=materials,
                transform=transform,
                unset_attributes_return_none=True)  # for greater backward compatibility
    finally:
        del stage, file_path_or_stage

    if return_list:
        return list(results.values())
    return results


def add_mesh(stage, scene_path, vertices=None, faces=None, uvs=None, face_uvs_idx=None, face_normals=None,
             material_assignments=None, materials=None, local_to_world=None, time=None, overwrite_textures=False):
    r"""Add a mesh to an existing USD stage. The stage is modified but not saved to disk; material textures are written
    to disk.

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
        local_to_world (torch.FloatTensor, optional): local-to-world transform of shape ``(4, 4)``.
        time (convertible to float, optional): Positive integer defining the time at which the supplied parameters
            correspond to.
        overwrite_textures (bool): set to True to overwrite existing image files when writing textures; if False
            (default) will add index to filename to avoid conflicts.

    Returns:
        (Usd.Prim): the generated mesh Prim.

    Example:
        >>> vertices = torch.rand(3, 3)
        >>> faces = torch.tensor([[0, 1, 2]])
        >>> stage = create_stage('./new_stage.usd')
        >>> prim = add_mesh(stage, '/World/mesh', vertices, faces)
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
        uvs_list = uvs.view(-1, 2).detach().cpu().float().numpy().copy()
        uvs_list[:, 1] = 1 - uvs_list[:, 1]  # reverse convention we use in the code
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
            face_idx = [int(x) for x in list((material_assignments == i).nonzero().reshape((-1,)).detach().cpu().numpy())]
            subset_prim = add_subset(stage, usd_mesh.GetPrim(), f'subset_{i}',
                                     torch.tensor(face_idx, dtype=torch.int64),
                                     family_name='materialBind')
            try:
                # Note: this has the desirable property of always rewriting the materials under scene_path, but
                # TODO: more thought is required on when it is desirable to overwrite textures vs. not;
                # e.g. when re-exporting to the same file overwriting might be good, when calling `add_mesh`
                # repeatedly with new meshes, the textures of new materials should not overwrite existing
                # materials. Current default ensures meshes can be added without overwriting each other's
                # material files, and when a mesh is re-added, its materials will be reset.
                basename = str(pathlib.Path(stage.GetRootLayer().realPath).stem)
                export_material(material, stage, bound_prims=[subset_prim], time=time,
                                scene_path=f'{scene_path}/Looks/material_{i}',
                                texture_path=f'{basename}_textures',
                                overwrite_textures=overwrite_textures)
            except Exception as e:
                raise(e)

    if local_to_world is not None:
        set_local_to_world_transform(stage, usd_mesh.GetPrim(), local_to_world, time)

    return usd_mesh.GetPrim()

def export_mesh(file_path, scene_path='/World/Meshes/mesh_0', vertices=None, faces=None,
                uvs=None, face_uvs_idx=None, face_normals=None, material_assignments=None, materials=None,
                local_to_world=None, up_axis='Y', time=None, overwrite_textures=False, overwrite=False):
    r"""Export a single mesh to USD and save the stage to disk.

    .. note::

        Since v0.18.0 this function can only overwrite existing file or raise an error. Use :func:`add_mesh` to modify existing usd file.

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
        local_to_world (torch.FloatTensor, optional): local-to-world transform of shape ``(4, 4)``.
        up_axis (str, optional): Specifies the scene's up axis. Choose from ``['Y', 'Z']``
        time (convertible to float, optional):
            Positive integer defining the time at which the supplied parameters correspond to.
        overwrite_textures (bool): Set to True to overwrite existing image files when writing textures; if False (default) will add index to filename to avoid conflicts.
        overwrite (bool): If True, overwrite existing .usda. If False (default) raise an error if files already exists.

    Example:
        >>> vertices = torch.rand(3, 3)
        >>> faces = torch.tensor([[0, 1, 2]])
        >>> export_mesh('./new_stage.usd', vertices=vertices, faces=faces)
    """
    assert isinstance(scene_path, str)
    if time is None:
        time = Usd.TimeCode.Default()
    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"{file_path} already exists; to overwrite whole file use 'overwrite' argument;" +
                              "to add mesh to existing usd, use add_mesh' instead.")
    stage = create_stage(file_path, up_axis)
    add_mesh(stage, scene_path, vertices, faces, uvs, face_uvs_idx,
             face_normals, material_assignments, materials, local_to_world=local_to_world,
             time=time, overwrite_textures=overwrite_textures)
    stage.Save()

def export_meshes(file_path, scene_paths=None, vertices=None, faces=None,
                  uvs=None, face_uvs_idx=None, face_normals=None, material_assignments=None, materials=None,
                  local_to_world=None, up_axis='Y', times=None, overwrite_textures=False, overwrite=False):
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
        local_to_world (torch.FloatTensor, optional): local-to-world transforms as a single ``(4, 4)`` tensor
            (broadcast to every mesh) or a batched ``(N, 4, 4)`` tensor (one transform per mesh). Aligned
            with :attr:`kaolin.rep.SurfaceMesh.transform`.
        up_axis (str, optional): Specifies the scene's up axis. Choose from ``['Y', 'Z']``.
        times (list of int, optional): Positive integers defining the time at which the supplied parameters
            correspond to.
        overwrite_textures (bool): set to True to overwrite existing image files when writing textures; if False
            (default) will add index to filename to avoid conflicts.
        overwrite (bool): If True, overwrite existing .usda. If False (default) raise an error if files already exists.

    Example:
        >>> vertices_list = [torch.rand(3, 3) for _ in range(3)]
        >>> faces_list = [torch.tensor([[0, 1, 2]]) for _ in range(3)]
        >>> export_meshes('./new_stage.usd', vertices=vertices_list, faces=faces_list)
    """
    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"{file_path} already exists")
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

    if local_to_world is not None:
        if not torch.is_tensor(local_to_world):
            raise TypeError(f'Unexpected type {type(local_to_world)} for local_to_world (torch.Tensor expected)')
        if local_to_world.ndim == 3:
            if num_meshes == -1:
                num_meshes = local_to_world.shape[0]
            assert local_to_world.shape == (num_meshes, 4, 4), (
                f'local_to_world batched tensor must have shape ({num_meshes}, 4, 4), '
                f'got {tuple(local_to_world.shape)}')
        elif local_to_world.ndim == 2:
            assert tuple(local_to_world.shape) == (4, 4), (
                f'local_to_world tensor must have shape (4, 4) or (N, 4, 4), '
                f'got {tuple(local_to_world.shape)}')
        else:
            raise ValueError(
                f'local_to_world tensor must have shape (4, 4) or (N, 4, 4), '
                f'got {tuple(local_to_world.shape)}')

    if scene_paths is None:
        if not stage.GetPrimAtPath('/World/Meshes'):
            stage.DefinePrim('/World/Meshes', 'Xform')
        scene_paths = [f'/World/Meshes/mesh_{i}' for i in range(len(vertices))]

    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)

    for i, scene_path in enumerate(tqdm(scene_paths, desc="Exporting to USD", unit="mesh")):
        if local_to_world is None:
            ltw_i = None
        elif local_to_world.ndim == 3:
            ltw_i = local_to_world[i]
        else:
            ltw_i = local_to_world
        # Note: we make parameters explicit to ensure tests catch any API changes reliably
        add_mesh(stage, scene_path,
                 vertices=None if vertices is None else vertices[i],
                 faces=None if faces is None else faces[i],
                 uvs=None if uvs is None else uvs[i],
                 face_uvs_idx=None if face_uvs_idx is None else face_uvs_idx[i],
                 face_normals=None if face_normals is None else face_normals[i],
                 material_assignments=None if material_assignments is None else material_assignments[i],
                 materials=None if materials is None else materials[i],
                 local_to_world=ltw_i,
                 time=times[i], overwrite_textures=overwrite_textures)
    stage.Save()
