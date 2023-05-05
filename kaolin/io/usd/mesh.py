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

from .utils import _get_stage_from_maybe_file, get_scene_paths, create_stage

mesh_return_type = namedtuple('mesh_return_type', ['vertices', 'faces',
                                                   'uvs', 'face_uvs_idx', 'face_normals', 'materials_order',
                                                   'materials'])
__all__ = [
    'import_mesh',
    'import_meshes',
    'add_mesh',
    'export_mesh',
    'export_meshes',
    'get_raw_mesh_prim_geometry',
    'NonHomogeneousMeshError',
    'heterogeneous_mesh_handler_skip',
    'heterogeneous_mesh_handler_empty',
    'heterogeneous_mesh_handler_naive_homogenize'
]


def get_uvmap_primvar(mesh_prim):
    primvars = UsdGeom.PrimvarsAPI(mesh_prim)
    mesh_st = primvars.GetPrimvar('st')

    if not mesh_st.IsDefined():
        for pv in primvars.GetPrimvars():
            if pv.GetTypeName() == Sdf.ValueTypeNames.TexCoord2fArray:
                mesh_st = pv

    # This seems to be true for blender exports
    if not mesh_st.IsDefined():
        mesh_st = primvars.GetPrimvar('UVMap')

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
    dictionary of some or all values:
        - **vertices** (torch.FloatTensor): vertex positions with any transforms already applied, of shape (N, 3)
        - **transform** (torch.FloatTensor): applied transform of shape (4, 4)
        - **faces** (torch.LongTensor): face vertex indices of original shape saved in the USD
        - **face_sizes** (torch.LongTensor): face vertex counts of original shape saved in the USD
        if `with_normals=True`:
        - **normals** (torch.FloatTensor): normal values of original shape saved in the USD
        - **normals_interpolation** (string): normal interpolation type saved in the USD, such as "faceVarying"
        if `with_uvs=True`:
        - **uvs** (torch.FloatTensor): raw UV values saved in the USD
        - **face_uvs_idx** (torch.LongTensor): raw indices into the UV for every vertex of every face
        - **uv_interpolation** (string): UV interpolation type saved in the USD, such as "faceVarying"
    """
    if time is None:
        time = Usd.TimeCode.Default()

    mesh = UsdGeom.Mesh(mesh_prim)

    # Vertices
    vertices = mesh.GetPointsAttr().Get(time=time)
    transform = torch.from_numpy(
        np.array(UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(time), dtype=np.float32))
    if vertices:
        vertices = torch.from_numpy(np.array(vertices, dtype=np.float32))
        vertices_homo = torch.nn.functional.pad(vertices, (0, 1), mode='constant', value=1.)
        vertices = (vertices_homo @ transform)[:, :3]

    # Faces
    face_sizes = mesh.GetFaceVertexCountsAttr().Get(time=time)
    if face_sizes:
        face_sizes = torch.from_numpy(np.array(face_sizes, dtype=np.int64))

    faces = mesh.GetFaceVertexIndicesAttr().Get(time=time)
    if faces:
        faces = torch.from_numpy(np.array(faces, dtype=np.int64))

    # Normals
    normals = None
    normals_interpolation = None
    if with_normals:
        normals = mesh.GetNormalsAttr().Get(time=time)
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
            uvs = mesh_st.Get(time=time)
            uv_idx = mesh_st.GetIndices(time=time)  # For faces or for vertices
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


def get_face_uvs_idx(faces, face_sizes, uvs, uv_idx, uv_interpolation):
    if uv_interpolation in ['vertex', 'varying']:
        if not uv_idx:
            # for vertex and varying interpolation, length of uv_idx should match
            # length of mesh_vertex_indices
            uv_idx = list(range(len(uvs)))
        uv_idx = torch.tensor(uv_idx)
        face_uvs_idx = uv_idx[faces]
    elif uv_interpolation == 'faceVarying':
        if not uv_idx:
            # for faceVarying interpolation, length of uv_idx should match
            # num_faces * face_size
            # TODO(mshugrina): implement default behaviour
            uv_idx = [i for i, c in enumerate(face_sizes) for _ in range(c)]
            uv_idx = [i for i in range(sum(face_sizes))]
            raise NotImplementedError(f'Interpolation {uv_interpolation} for UV '
                                      'not supported when no uv indices provided. '
                                      'File a bug to fix.')
        face_uvs_idx = uv_idx
    # TODO: implement uniform interpolation
    # elif mesh_uv_interpolation == 'uniform':
    else:
        raise NotImplementedError(f'Interpolation type {uv_interpolation} is '
                                  'not supported')
    return face_uvs_idx


def _get_flattened_mesh_attributes(stage, scene_path, with_materials, with_normals, time):
    """Return mesh attributes flattened into a single mesh."""
    stage_dir = os.path.dirname(str(stage.GetRootLayer().realPath))
    prim = stage.GetPrimAtPath(scene_path)
    if not prim:
        raise ValueError(f'No prim found at "{scene_path}".')

    attrs = {}

    def _process_mesh_prim(mesh_prim, ref_path, attrs, time):
        # TODO: use helper methods here for: get_raw_mesh_prim_geometry, get_face_uvs_idx, get_face_normals
        cur_first_idx_faces = sum([len(v) for v in attrs.get('vertices', [])])
        cur_first_idx_uvs = sum([len(u) for u in attrs.get('uvs', [])])
        mesh = UsdGeom.Mesh(mesh_prim)
        mesh_vertices = mesh.GetPointsAttr().Get(time=time)
        mesh_face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get(time=time)
        mesh_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get(time=time)
        mesh_st = get_uvmap_primvar(mesh_prim)
        mesh_subsets = UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(mesh_prim))
        mesh_material = UsdShade.MaterialBindingAPI.Apply(mesh_prim).ComputeBoundMaterial()[0]
        transform = torch.from_numpy(np.array(
            UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(time), dtype=np.float32
        ))

        # Parse mesh UVs
        if mesh_st:
            mesh_uvs = mesh_st.Get(time=time)
            mesh_uv_indices = mesh_st.GetIndices(time=time)
            mesh_uv_interpolation = mesh_st.GetInterpolation()
        mesh_face_normals = mesh.GetNormalsAttr().Get(time=time)

        # Parse mesh geometry
        if mesh_vertices:
            mesh_vertices = torch.from_numpy(np.array(mesh_vertices, dtype=np.float32))
            mesh_vertices_homo = torch.nn.functional.pad(mesh_vertices, (0, 1), mode='constant', value=1.)
            mesh_vertices = (mesh_vertices_homo @ transform)[:, :3]
            attrs.setdefault('vertices', []).append(mesh_vertices)
        if mesh_vertex_indices:
            attrs.setdefault('face_vertex_counts', []).append(torch.from_numpy(
                np.array(mesh_face_vertex_counts, dtype=np.int64)))
            vertex_indices = torch.from_numpy(np.array(mesh_vertex_indices, dtype=np.int64)) + cur_first_idx_faces
            attrs.setdefault('vertex_indices', []).append(vertex_indices)
        if with_normals and mesh_face_normals:
            mesh_face_normals = torch.from_numpy(np.array(mesh_face_normals, dtype=np.float32))
            mesh_face_normals_homo = torch.nn.functional.pad(mesh_face_normals, (0, 1), mode='constant', value=1.)
            mesh_face_normals = (mesh_face_normals_homo @ transform)[:, :3]
            attrs.setdefault('face_normals', []).append(mesh_face_normals)
        if mesh_st and mesh_uvs:
            attrs.setdefault('uvs', []).append(torch.from_numpy(np.array(mesh_uvs, dtype=np.float32)))
            if mesh_uv_interpolation in ['vertex', 'varying']:
                if not mesh_uv_indices:
                    # for vertex and varying interpolation, length of mesh_uv_indices should match
                    # length of mesh_vertex_indices
                    mesh_uv_indices = list(range(len(mesh_uvs)))
                mesh_uv_indices = torch.tensor(mesh_uv_indices) + cur_first_idx_uvs
                face_uvs_idx = mesh_uv_indices[torch.from_numpy(np.array(mesh_vertex_indices, dtype=np.int64))]
                attrs.setdefault('face_uvs_idx', []).append(face_uvs_idx)
            elif mesh_uv_interpolation == 'faceVarying':
                if not mesh_uv_indices:
                    # for faceVarying interpolation, length of mesh_uv_indices should match
                    # num_faces * face_size

                    mesh_uv_indices = [i for i in range(sum(mesh_face_vertex_counts))]
                attrs.setdefault('face_uvs_idx', []).append(torch.tensor(mesh_uv_indices) + cur_first_idx_uvs)
            # elif mesh_uv_interpolation == 'uniform':
            # TODO: refactor to use get_face_uvs_idx
            else:
                raise NotImplementedError(f'Interpolation type {mesh_uv_interpolation} is '
                                          'not currently supported')

        # Parse mesh materials
        if with_materials:
            subset_idx_map = {}
            attrs.setdefault('materials', [])
            attrs.setdefault('material_idx_map', {})
            if mesh_material:
                mesh_material_path = str(mesh_material.GetPath())
                if mesh_material_path in attrs['material_idx_map']:
                    material_idx = attrs['material_idx_map'][mesh_material_path]
                else:
                    try:
                        material = usd_materials.MaterialManager.read_usd_material(
                            mesh_material, stage_dir, time)
                        material_idx = len(attrs['materials'])
                        attrs['materials'].append(material)
                        attrs['material_idx_map'][mesh_material_path] = material_idx
                    except usd_materials.MaterialNotSupportedError as e:
                        warnings.warn(e.args[0])
                    except usd_materials.MaterialLoadError as e:
                        warnings.warn(e.args[0])
            if mesh_subsets:
                for subset in mesh_subsets:
                    subset_material, _ = UsdShade.MaterialBindingAPI.Apply(
                        subset.GetPrim()
                    ).ComputeBoundMaterial()
                    subset_material_metadata = subset_material.GetPrim().GetMetadata('references')
                    mat_ref_path = ""
                    if ref_path:
                        mat_ref_path = ref_path
                    if subset_material_metadata:
                        asset_path = subset_material_metadata.GetAddedOrExplicitItems()[0].assetPath
                        mat_ref_path = os.path.join(ref_path, os.path.dirname(asset_path))
                    if not os.path.isabs(mat_ref_path):
                        mat_ref_path = os.path.join(stage_dir, mat_ref_path)
                    try:
                        kal_material = usd_materials.MaterialManager.read_usd_material(
                            subset_material, mat_ref_path, time)
                    except usd_materials.MaterialNotSupportedError as e:
                        warnings.warn(e.args[0])
                        continue
                    except usd_materials.MaterialLoadError as e:
                        warnings.warn(e.args[0])

                    subset_material_path = str(subset_material.GetPath())
                    if subset_material_path not in attrs['material_idx_map']:
                        attrs['material_idx_map'][subset_material_path] = len(attrs['materials'])
                        attrs['materials'].append(kal_material)
                    subset_indices = np.array(subset.GetIndicesAttr().Get())
                    subset_idx_map[attrs['material_idx_map'][subset_material_path]] = subset_indices
            # Create material face index list
            if mesh_face_vertex_counts:
                # TODO: this is inefficient; include in the refactor of USD code
                for face_idx in range(len(mesh_face_vertex_counts)):
                    is_in_subsets = False
                    for subset_idx in subset_idx_map:
                        if face_idx in subset_idx_map[subset_idx]:
                            is_in_subsets = True
                            attrs.setdefault('materials_face_idx', []).extend(
                                [subset_idx] * mesh_face_vertex_counts[face_idx]
                            )
                    if not is_in_subsets:
                        if mesh_material:
                            attrs.setdefault('materials_face_idx', []).extend(
                                [material_idx] * mesh_face_vertex_counts[face_idx]
                            )
                        else:
                            # Assign to `None` material (ie. index 0)
                            attrs.setdefault('materials_face_idx', []).extend([0] * mesh_face_vertex_counts[face_idx])

    def _traverse(cur_prim, ref_path, attrs, time):
        metadata = cur_prim.GetMetadata('references')
        if metadata:
            ref_path = os.path.dirname(metadata.GetAddedOrExplicitItems()[0].assetPath)
        if UsdGeom.Mesh(cur_prim):
            _process_mesh_prim(cur_prim, ref_path, attrs, time)
        for child in cur_prim.GetChildren():
            _traverse(child, ref_path, attrs, time)

    _traverse(stage.GetPrimAtPath(scene_path), '', attrs, time)

    if not attrs.get('vertices'):
        warnings.warn(f'Scene object at {scene_path} contains no vertices.', UserWarning)

    # Only import vertices if they are defined for the entire mesh
    if all([v is not None for v in attrs.get('vertices', [])]) and len(attrs.get('vertices', [])) > 0:
        attrs['vertices'] = torch.cat(attrs.get('vertices'))
    else:
        attrs['vertices'] = None
    # Only import vertex index and counts if they are defined for the entire mesh
    if all([vi is not None for vi in attrs.get('vertex_indices', [])]) and len(attrs.get('vertex_indices', [])) > 0:
        attrs['face_vertex_counts'] = torch.cat(attrs.get('face_vertex_counts', []))
        attrs['vertex_indices'] = torch.cat(attrs.get('vertex_indices', []))
    else:
        attrs['face_vertex_counts'] = None
        attrs['vertex_indices'] = None
    # Only import UVs if they are defined for the entire mesh
    if not all([uv is not None for uv in attrs.get('uvs', [])]) or len(attrs.get('uvs', [])) == 0:
        if len(attrs.get('uvs', [])) > 0:
            warnings.warn('UVs are missing for some child meshes for prim at '
                          f'{scene_path}. As a result, no UVs were imported.')
        attrs['uvs'] = None
        attrs['face_uvs_idx'] = None
    else:
        attrs['uvs'] = torch.cat(attrs['uvs'])
        if attrs.get('face_uvs_idx', None):
            attrs['face_uvs_idx'] = torch.cat(attrs['face_uvs_idx'])
        else:
            attrs['face_uvs_idx'] = None

    # Only import face_normals if they are defined for the entire mesh
    if not all([n is not None for n in attrs.get('face_normals', [])]) or len(attrs.get('face_normals', [])) == 0:
        if len(attrs.get('face_normals', [])) > 0:
            warnings.warn('Face normals are missing for some child meshes for '
                          f'prim at {scene_path}. As a result, no Face Normals were imported.')
        attrs['face_normals'] = None
    else:
        attrs['face_normals'] = torch.cat(attrs['face_normals'])

    if attrs.get('materials_face_idx') is None:
        attrs['materials_face_idx'] = None
    else:
        attrs['materials_face_idx'] = torch.LongTensor(attrs['materials_face_idx'])

    if all([m is None for m in attrs.get('materials', [])]):
        attrs['materials'] = None
    return attrs


def import_mesh(file_path_or_stage, scene_path=None, with_materials=False, with_normals=False,
                heterogeneous_mesh_handler=None, time=None):
    r"""Import a single mesh from a USD file of Stage in an unbatched representation.

    Supports homogeneous meshes (meshes with consistent numbers of vertices per face).
    All sub-meshes found under the `scene_path` are flattened to a single mesh. The following
    interpolation types are supported for UV coordinates: `vertex`, `varying` and `faceVarying`.
    Returns an unbatched representation.

    Args:
        file_path_or_stage (str, Usd.Stage):
            Path to usd file (`\*.usd`, `\*.usda`) or :class:`Usd.Stage`.
        scene_path (str, optional): Scene path within the USD file indicating which primitive to import.
            If not specified, the all meshes in the scene will be imported and flattened into a single mesh.
        with_materials (bool): if True, load materials. Default: False.
        with_normals (bool): if True, load vertex normals. Default: False.
        heterogeneous_mesh_handler (function, optional): Optional function to handle heterogeneous meshes.
            The function's input and output must be  ``vertices`` (torch.FloatTensor), ``faces`` (torch.LongTensor),
            ``uvs`` (torch.FloatTensor), ``face_uvs_idx`` (torch.LongTensor), and ``face_normals`` (torch.FloatTensor).
            If the function returns ``None``, the mesh will be skipped. If no function is specified,
            an error will be raised when attempting to import a heterogeneous mesh.
        time (convertible to float, optional): Positive integer indicating the time at which to retrieve parameters.

    Returns:

    namedtuple of:
        - **vertices** (torch.FloatTensor): of shape (num_vertices, 3)
        - **faces** (torch.LongTensor): of shape (num_faces, face_size)
        - **uvs** (torch.FloatTensor): of shape (num_uvs, 2)
        - **face_uvs_idx** (torch.LongTensor): of shape (num_faces, face_size)
        - **face_normals** (torch.FloatTensor): of shape (num_faces, face_size, 3)
        - **materials** (list of kaolin.io.materials.Material): Material properties

    Example:
        >>> # Create a stage with some meshes
        >>> stage = export_mesh('./new_stage.usd', vertices=torch.rand(3, 3), faces=torch.tensor([[0, 1, 2]]),
        ... scene_path='/World/mesh1')
        >>> # Import meshes
        >>> mesh = import_mesh('./new_stage.usd', scene_path='/World/mesh1')
        >>> mesh.vertices.shape
        torch.Size([3, 3])
        >>> mesh.faces
        tensor([[0, 1, 2]])
    """
    # TODO  add arguments to selectively import UVs and normals
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    if scene_path is None:
        scene_path = stage.GetPseudoRoot().GetPath()
    if time is None:
        time = Usd.TimeCode.Default()
    meshes_list = import_meshes(stage, [scene_path],
                                heterogeneous_mesh_handler=heterogeneous_mesh_handler,
                                with_materials=with_materials,
                                with_normals=with_normals, times=[time])
    return mesh_return_type(*meshes_list[0])


def import_meshes(file_path_or_stage, scene_paths=None, with_materials=False, with_normals=False,
                  heterogeneous_mesh_handler=None, times=None):
    r"""Import one or more meshes from a USD file or Stage in an unbatched representation.

    Supports homogeneous meshes (meshes with consistent numbers of vertices per face). Custom handling of
    heterogeneous meshes can be achieved by passing a function through the ``heterogeneous_mesh_handler`` argument.
    The following interpolation types are supported for UV coordinates: `vertex`, `varying` and `faceVarying`.
    For each scene path specified in `scene_paths`, sub-meshes (if any) are flattened to a single mesh.
    Returns unbatched meshes list representation. Prims with no meshes or with heterogenous faces are skipped.

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (`\*.usd`, `\*.usda`) or :class:`Usd.Stage`.
        scene_paths (list of str, optional): Scene path(s) within the USD file indicating which primitive(s)
            to import. If None, all prims of type `Mesh` will be imported.
        with_materials (bool): if True, load materials. Default: False.
        with_normals (bool): if True, load vertex normals. Default: False.
        heterogeneous_mesh_handler (function, optional): Optional function to handle heterogeneous meshes.
            The function's input and output must be  ``vertices`` (torch.FloatTensor), ``faces`` (torch.LongTensor),
            ``uvs`` (torch.FloatTensor), ``face_uvs_idx`` (torch.LongTensor), and ``face_normals`` (torch.FloatTensor).
            If the function returns ``None``, the mesh will be skipped. If no function is specified,
            an error will be raised when attempting to import a heterogeneous mesh.
        times (list of int): Positive integers indicating the time at which to retrieve parameters.
    Returns:

    list of namedtuple of:
        - **vertices** (list of torch.FloatTensor): of shape (num_vertices, 3)
        - **faces** (list of torch.LongTensor): of shape (num_faces, face_size)
        - **uvs** (list of torch.FloatTensor): of shape (num_uvs, 2)
        - **face_uvs_idx** (list of torch.LongTensor): of shape (num_faces, face_size)
        - **face_normals** (list of torch.FloatTensor): of shape (num_faces, face_size, 3)
        - **materials** (list of kaolin.io.materials.Material): Material properties

    Example:
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
    """
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

    vertices_list, faces_list, uvs_list, face_uvs_idx_list, face_normals_list = [], [], [], [], []
    materials_order_list, materials_list = [], []
    silence_tqdm = len(scene_paths) < 10  # Silence tqdm if fewer than 10 paths are found
    for scene_path, time in zip(tqdm(scene_paths, desc="Importing from USD", unit="mesh", disable=silence_tqdm), times):
        mesh_attr = _get_flattened_mesh_attributes(stage, scene_path, with_materials, with_normals, time=time)
        #print("mesh_attr:", mesh_attr)
        vertices = mesh_attr['vertices']
        face_vertex_counts = mesh_attr['face_vertex_counts']
        faces = mesh_attr['vertex_indices']
        uvs = mesh_attr['uvs']
        face_uvs_idx = mesh_attr['face_uvs_idx']
        face_normals = mesh_attr['face_normals']
        materials_face_idx = mesh_attr['materials_face_idx']
        materials = mesh_attr['materials']
        materials_order = None
        # TODO(jlafleche) Replace tuple output with mesh class

        # Handle attributes that require faces
        if faces is not None:
            facesize = 0
            if face_vertex_counts is not None and face_vertex_counts.shape[0] > 0:
                facesize = face_vertex_counts[0]

            if not torch.all(face_vertex_counts == facesize):
                if heterogeneous_mesh_handler is None:
                    raise utils.NonHomogeneousMeshError(
                        f'Mesh at {scene_path} is non-homogeneous '
                        f'and cannot be imported from {repr(file_path_or_stage)}.')
                else:
                    mesh = heterogeneous_mesh_handler(vertices, face_vertex_counts, faces, uvs,
                                                      face_uvs_idx, face_normals, materials_face_idx)
                    if mesh is None:
                        continue
                    else:
                        vertices, face_vertex_counts, faces, uvs, face_uvs_idx, face_normals, materials_face_idx = mesh
                        facesize = faces.shape[-1]

            faces = faces.view(-1, facesize)  # Nfaces x facesize
            nfaces = faces.shape[0]
            if nfaces > 0:
                if face_uvs_idx is not None and face_uvs_idx.size(0) > 0:
                    uvs = uvs.reshape(-1, 2)
                    face_uvs_idx = face_uvs_idx.reshape(-1, facesize)
                    # TODO: should set to None otherwise for this and others?
                if face_normals is not None and face_normals.size(0) > 0:
                    face_normals = face_normals.reshape(nfaces, -1, 3)

                if materials_face_idx is not None:            # Create material order list
                    materials_face_idx = materials_face_idx.view(-1, facesize)
                    cur_mat_idx = -1
                    materials_order = []
                    for idx in range(len(materials_face_idx)):
                        mat_idx = materials_face_idx[idx][0].item()
                        if cur_mat_idx != mat_idx:
                            cur_mat_idx = mat_idx
                            materials_order.append([idx, mat_idx])

        vertices_list.append(vertices)
        faces_list.append(faces)
        uvs_list.append(uvs)
        face_uvs_idx_list.append(face_uvs_idx)
        face_normals_list.append(face_normals)
        materials_order_list.append(materials_order)
        materials_list.append(materials)

    params = [vertices_list, faces_list, uvs_list, face_uvs_idx_list,
              face_normals_list, materials_order_list, materials_list]
    return [mesh_return_type(v, f, uv, fuv, fn, mo, m) for v, f, uv, fuv, fn, mo, m in zip(*params)]

def add_mesh(stage, scene_path, vertices=None, faces=None, uvs=None, face_uvs_idx=None, face_normals=None,
             materials_order=None, materials=None, time=None):
    r"""Add a mesh to an existing USD stage.

    Add a mesh to the USD stage. The stage is modified but not saved to disk.

    Args:
        stage (Usd.Stage): Stage onto which to add the mesh.
        scene_path (str): Absolute path of mesh within the USD file scene. Must be a valid ``Sdf.Path``.
        vertices (torch.FloatTensor, optional): Vertices with shape ``(num_vertices, 3)``.
        faces (torch.LongTensor, optional): Vertex indices for each face with shape ``(num_faces, face_size)``.
            Mesh must be homogenous (consistent number of vertices per face).
        uvs (torch.FloatTensor, optional): of shape ``(num_uvs, 2)``.
        face_uvs_idx (torch.LongTensor, optional): of shape ``(num_faces, face_size)``. If provided, `uvs` must also
            be specified.
        face_normals (torch.Tensor, optional): of shape ``(num_faces, face_size, 3)``.
        materials_order (torch.LongTensor): of shape (N, 2)
          showing the order in which materials are used over **face_uvs_idx** and the first indices
          in which they start to be used. A material can be used multiple times.
        materials (list of Material): a list of materials
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

    if faces is not None and materials_order is not None and materials is not None:
        stage.DefinePrim(f'{scene_path}/Looks', 'Scope')
        subsets = {}
        for i in range(len(materials_order)):
            first_face_idx, mat_idx = materials_order[i]
            if materials[mat_idx] is None:
                continue
            last_face_idx = materials_order[i + 1][0] if (i + 1) < len(materials_order) else faces.size(0)
            for face_idx in range(first_face_idx, last_face_idx):
                subsets.setdefault(mat_idx, []).append(face_idx)

        # Create submeshes
        for i, subset in enumerate(subsets):
            subset_prim = stage.DefinePrim(f'{scene_path}/subset_{i}', 'GeomSubset')
            subset_prim.GetAttribute('indices').Set(subsets[subset])
            if isinstance(materials[subset], usd_materials.Material):
                materials[subset]._write_usd_preview_surface(stage, f'{scene_path}/Looks/material_{subset}',
                                                             [subset_prim], time, texture_dir=f'material_{subset}',
                                                             texture_file_prefix='')    # TODO allow users to pass root path to save textures to

    return usd_mesh.GetPrim()


def export_mesh(file_path, scene_path='/World/Meshes/mesh_0', vertices=None, faces=None,
                uvs=None, face_uvs_idx=None, face_normals=None, materials_order=None, materials=None,
                up_axis='Y', time=None):
    r"""Export a single mesh to USD.

    Export a single mesh defined by vertices and faces and save the stage to disk.

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
        materials_order (torch.LongTensor):
          of shape (N, 2),
          showing the order in which materials are used over **face_uvs_idx** and the first indices
          in which they start to be used. A material can be used multiple times.
        materials (list of Material): a list of materials
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
             face_normals, materials_order, materials, time=time)
    stage.Save()

    return stage

def export_meshes(file_path, scene_paths=None, vertices=None, faces=None,
                  uvs=None, face_uvs_idx=None, face_normals=None, materials_order=None, materials=None,
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
        materials_order (torch.LongTensor): of shape (N, 2)
          showing the order in which materials are used over **face_uvs_idx** and the first indices
          in which they start to be used. A material can be used multiple times.
        materials (list of Material): a list of materials
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
    mesh_parameters = {'vertices': vertices, 'faces': faces, 'uvs': uvs,
                       'face_uvs_idx': face_uvs_idx, 'face_normals': face_normals,
                       'materials_order': materials_order, 'materials': materials}
    supplied_parameters = {k: p for k, p in mesh_parameters.items() if p is not None}
    length = len(list(supplied_parameters.values())[0])
    assert all([len(p) == length for p in supplied_parameters.values()])
    if scene_paths is None:
        if not stage.GetPrimAtPath('/World/Meshes'):
            stage.DefinePrim('/World/Meshes', 'Xform')
        scene_paths = [f'/World/Meshes/mesh_{i}' for i in range(len(vertices))]
    assert len(scene_paths) == length
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)

    for i, scene_path in enumerate(tqdm(scene_paths, desc="Exporting to USD", unit="mesh")):
        mesh_params = {k: p[i] for k, p in supplied_parameters.items()}
        add_mesh(stage, scene_path, **mesh_params)
    stage.Save()

    return stage

####################################################
#                      ERRORS                      #
####################################################

class NonHomogeneousMeshError(Exception):
    """Raised when expecting a homogeneous mesh but a heterogenous
    mesh is encountered.

    .. deprecated:: 1.11.0
       This function is deprecated, use :func:`kaolin.io.utils.NonHomogeneousMeshError`.

    Attributes:
        message (str)
    """
    __slots__ = ['message']

    def __init__(self, message):
        warnings.warn("NonHomogeneousMeshError is deprecated, "
                      "please use kaolin.io.utils.NonHomogeneousMeshError instead",
                      DeprecationWarning, stacklevel=2)
        self.message = message

####################################################
#                  ERROR HANDLERS                  #
####################################################

# Mesh Functions
# TODO(jlafleche) Support mesh subgroups for materials
def heterogeneous_mesh_handler_skip(*args):
    r"""Skip heterogeneous meshes.

    .. deprecated:: 1.11.0
       This function is deprecated, use :func:`kaolin.io.utils.heterogeneous_mesh_handler_skip`.

    """
    warnings.warn("heterogeneous_mesh_handler_skip is deprecated, "
                  "please use kaolin.io.utils.heterogeneous_mesh_handler_skip instead",
                  DeprecationWarning, stacklevel=2)

    return None

def heterogeneous_mesh_handler_empty(*args):
    """Return empty tensors for vertices and faces of heterogeneous meshes.

    .. deprecated:: 1.11.0
       This function is deprecated, use :func:`kaolin.io.utils.heterogeneous_mesh_handler_empty`.

    """
    warnings.warn("heterogeneous_mesh_handler_empty is deprecated, "
                  "please use kaolin.io.utils.heterogeneous_mesh_handler_empty instead",
                  DeprecationWarning, stacklevel=2)

    return (torch.FloatTensor(size=(0, 3)), torch.LongTensor(size=(0,)),
            torch.LongTensor(size=(0, 3)), torch.FloatTensor(size=(0, 2)),
            torch.LongTensor(size=(0, 3)), torch.FloatTensor(size=(0, 3, 3)),
            torch.LongTensor(size=(0,)))

def heterogeneous_mesh_handler_naive_homogenize(vertices, face_vertex_counts, *args):
    r"""Homogenize list of faces containing polygons of varying number of edges to triangles using fan
    triangulation.

    .. deprecated:: 1.11.0
       This function is deprecated, use :func:`kaolin.io.utils.heterogeneous_mesh_handler_naive_homogenize`.

    Args:
        vertices (torch.FloatTensor): Vertices with shape ``(N, 3)``.
        face_vertex_counts (torch.LongTensor): Number of vertices for each face with shape ``(M)``
            for ``M`` faces.
        *args: Variable length features that need to be handled. For example, faces and uvs.

    Returns:
        (list of torch.tensor): Homogeneous list of attributes.
    """

    warnings.warn("heterogeneous_mesh_handler_naive_homogenize is deprecated, "
                  "please use kaolin.io.utils.heterogeneous_mesh_handler_naive_homogenize instead",
                  DeprecationWarning, stacklevel=2)

    def _homogenize(attr, face_vertex_counts):
        if attr is not None:
            attr = attr.tolist()
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

    new_attrs = [_homogenize(a, face_vertex_counts) for a in args]
    new_counts = torch.ones(vertices.size(0), dtype=torch.long).fill_(3)
    return (vertices, new_counts, *new_attrs)
