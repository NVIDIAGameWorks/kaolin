# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

"""Export common 3D representations to USD format. Mesh (homogeneous), Voxelgrid
and PointCloud representations are currently supported.
"""

import os
import re
import warnings
from collections import namedtuple

import torch

try:
    from pxr import Usd, UsdGeom, Vt, Sdf
except ImportError:
    warnings.warn("Warning: module pxr not found", ImportWarning)



mesh_return_type = namedtuple('mesh_return_type', ['vertices', 'faces', 'uvs', 'face_uvs_idx', 'face_normals', 'materials'])


class NonHomogeneousMeshError(Exception):
    """Raised when expecting a homogeneous mesh but a heterogenous
    mesh is encountered.

    Attributes:
        message (str)
    """
    __slots__ = ['message']

    def __init__(self, message):
        self.message = message


def _get_stage_next_free_path(stage, scene_path):
    """Get next free path in the stage."""
    default_prim = stage.GetDefaultPrim()
    if default_prim:
        scene_path = default_prim.GetPath().pathString + scene_path

    match = re.search(r'_(\d+)$', scene_path)
    i = int(match.group(1)) if match else 0

    while stage.GetPrimAtPath(scene_path):
        scene_path = f'{scene_path}_{i:02d}'

    return scene_path


def _get_flattened_mesh_attributes(stage, scene_path, time):
    """Return mesh attributes flattened into a single mesh."""
    prim = stage.GetPrimAtPath(scene_path)
    if not prim:
        raise ValueError(f'No prim found at "{scene_path}".')
    mesh_prims = [x for x in Usd.PrimRange(prim).AllPrims(prim) if UsdGeom.Mesh(x)]
    cur_first_idx_faces = 0
    cur_first_idx_uvs = 0
    vertices, vertex_indices, face_vertex_counts, uvs, face_uvs_idx, face_normals, materials = [], [], [], [], [], [], []
    for mesh_prim in mesh_prims:
        mesh = UsdGeom.Mesh(mesh_prim)
        mesh_vertices = mesh.GetPointsAttr().Get(time=time)
        mesh_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get(time=time)
        mesh_st = mesh.GetPrimvar('st')
        if mesh_st:
            mesh_uvs = mesh_st.Get(time=time)
            mesh_uv_indices = mesh_st.GetIndices(time=time)
            mesh_uv_interpolation = mesh_st.GetInterpolation()
        mesh_face_normals = mesh.GetNormalsAttr().Get(time=time)
        if mesh_vertices:
            vertices.append(torch.tensor(mesh_vertices))
        if mesh_vertex_indices:
            face_vertex_counts.append(torch.tensor(mesh.GetFaceVertexCountsAttr().Get(time=time)))
            vertex_indices.append(torch.tensor(mesh_vertex_indices) + cur_first_idx_faces)
            if vertices:
                cur_first_idx_faces += len(vertices[-1])
        if mesh_face_normals:
            face_normals.append(torch.tensor(mesh_face_normals))
        if mesh_st and mesh_uvs:
            uvs.append(torch.tensor(mesh_uvs))
            if mesh_uv_interpolation in ['vertex', 'varying']:
                if not mesh_uv_indices:
                    # for vertex and varying interpolation, length of mesh_uv_indices should match
                    # length of mesh_vertex_indices
                    mesh_uv_indices = list(range(len(mesh_uvs)))
                mesh_uv_indices = torch.tensor(mesh_uv_indices) + cur_first_idx_uvs
                face_uvs_idx.append(mesh_uv_indices[torch.tensor(mesh_vertex_indices)])
            elif mesh_uv_interpolation == 'faceVarying':
                if not mesh_uv_indices:
                    # for faceVarying interpolation, length of mesh_uv_indices should match
                    # num_faces * face_size
                    mesh_uv_indices = list(range(len(mesh_uvs)))
                face_uvs_idx.append(torch.tensor(mesh_uv_indices) + cur_first_idx_uvs)
            else:
                raise NotImplementedError(f'Interpolation type {mesh_uv_interpolation} is '
                                          'not currently supported')
            cur_first_idx_uvs += len(mesh_uvs)
    if not vertices:
        warnings.warn(f'Scene object at {scene_path} contains no vertices.', UserWarning)

    # Only import vertices if they are defined for the entire mesh
    if all([v is not None for v in vertices]) and len(vertices) > 0:
        vertices = torch.cat(vertices)
    else:
        vertices = None
    # Only import vertex index and counts if they are defined for the entire mesh
    if all([vi is not None for vi in vertex_indices]) and len(vertex_indices) > 0:
        face_vertex_counts = torch.cat(face_vertex_counts)
        vertex_indices = torch.cat(vertex_indices)
    else:
        face_vertex_counts = None
        vertex_indices = None
    # Only import UVs if they are defined for the entire mesh
    if not all([uv is not None for uv in uvs]) or len(uvs) == 0:
        if len(uvs) > 0:
            warnings.warn('UVs are missing for some child meshes for prim at '
                          f'{scene_path}. As a result, no UVs were imported.')
        uvs = None
        face_uvs_idx = None
    else:
        uvs = torch.cat(uvs)
        face_uvs_idx = torch.cat(face_uvs_idx)

    # Only import face_normals if they are defined for the entire mesh
    if not all([n is not None for n in face_normals]) or len(face_normals) == 0:
        if len(face_normals) > 0:
            warnings.warn('Face normals are missing for some child meshes for '
                          f'prim at {scene_path}. As a result, no Face Normals were imported.')
        face_normals = None
    else:
        face_normals = torch.cat(face_normals)

    return vertices, face_vertex_counts, vertex_indices, uvs, face_uvs_idx, face_normals, materials


def get_root(file_path):
    r"""Return the root prim scene path.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).

    Returns:
        (str): Root scene path.

    Example:
        >>> # Create a stage with some meshes
        >>> vertices_list = [torch.rand(3, 3) for _ in range(3)]
        >>> faces_list = [torch.tensor([[0, 1, 2]]) for _ in range(3)]
        >>> stage = export_meshes('./new_stage.usd', vertices=vertices_list, faces=faces_list)
        >>> # Retrieve root scene path
        >>> root_prim = get_root('./new_stage.usd')
        >>> mesh = import_mesh('./new_stage.usd', root_prim)
        >>> mesh.vertices.shape
        torch.Size([9, 3])
        >>> mesh.faces.shape
        torch.Size([3, 3])
    """
    stage = Usd.Stage.Open(file_path)
    return stage.GetPseudoRoot().GetPath()


def get_scene_paths(file_path, scene_path_regex=None, prim_types=None):
    r"""Return all scene paths contained in specified file. Filter paths with regular
    expression in `scene_path_regex` if provided.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_path_regex (str, optional): Optional regular expression used to select returned scene paths.
        prim_types (list of str, optional): Optional list of valid USD Prim types used to
            select scene paths.

    Returns:
        (list of str): List of filtered scene paths.

    Example:
        >>> # Create a stage with some meshes
        >>> vertices_list = [torch.rand(3, 3) for _ in range(3)]
        >>> faces_list = [torch.tensor([[0, 1, 2]]) for _ in range(3)]
        >>> stage = export_meshes('./new_stage.usd', vertices=vertices_list, faces=faces_list)
        >>> # Retrieve scene paths
        >>> get_scene_paths('./new_stage.usd', prim_types=['Mesh'])
        [Sdf.Path('/World/Meshes/mesh_0'), Sdf.Path('/World/Meshes/mesh_1'), Sdf.Path('/World/Meshes/mesh_2')]
        >>> get_scene_paths('./new_stage.usd', scene_path_regex=r'.*_0', prim_types=['Mesh'])
        [Sdf.Path('/World/Meshes/mesh_0')]
    """
    stage = Usd.Stage.Open(file_path)
    if scene_path_regex is None:
        scene_path_regex = '.*'
    if prim_types is not None:
        prim_types = [pt.lower() for pt in prim_types]

    scene_paths = []
    for p in stage.Traverse():
        is_valid_prim_type = prim_types is None or p.GetTypeName().lower() in prim_types
        is_valid_scene_path = re.match(scene_path_regex, str(p.GetPath()))
        if is_valid_prim_type and is_valid_scene_path:
            scene_paths.append(p.GetPath())
    return scene_paths


def create_stage(file_path, up_axis='Y'):
    r"""Create a new USD file and return an empty stage.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        up_axis (['Y', 'Z']): Specify the stage up axis. Choose from ``['Y', 'Z']``.

    Returns:
        (Usd.Stage)

    Example:
        >>> stage = create_stage('./new_stage.usd', up_axis='Z')
        >>> type(stage)
        <class 'pxr.Usd.Stage'>
    """
    assert os.path.exists(os.path.dirname(file_path)), f'Directory {os.path.dirname(file_path)} not found.'
    stage = Usd.Stage.CreateNew(str(file_path))
    world = stage.DefinePrim('/World', 'Xform')
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageUpAxis(stage, up_axis)
    return stage


# Mesh Functions
# TODO(jlafleche) Support mesh subgroups for materials
def heterogeneous_mesh_handler_skip(*args):
    r"""Skip heterogeneous meshes."""
    return None


def heterogeneous_mesh_handler_empty(vertices, face_vertex_counts, faces, uvs, face_uvs_idx, face_normals):
    """Return empty tensors for vertices and faces of heterogeneous meshes."""
    return (torch.FloatTensor(size=(0, 3)), torch.LongTensor(size=(0,)),
            torch.LongTensor(size=(0, 3)), torch.FloatTensor(size=(0, 2)),
            torch.LongTensor(size=(0, 3)), torch.FloatTensor(size=(0, 3, 3)))


def heterogeneous_mesh_handler_naive_homogenize(vertices, face_vertex_counts, *args):
    r"""Homogenize list of faces containing polygons of varying number of edges to triangles using fan
    triangulation.

    Args:
        vertices (torch.FloatTensor): Vertices with shape ``(N, 3)``.
        face_vertex_counts (torch.LongTensor): Number of vertices for each face with shape ``(M)``
            for ``M`` faces.
        faces (torch.LongTensor): Vertex indices for each face of with shape ``(F)`` where ``F`` is
            the sum of ``face_vertex_counts``.

    Returns:
        (list of list of int): Homogeneous list of faces.
    """
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
            attr = torch.tensor(new_attr)
        return attr

    new_attrs = [_homogenize(a, face_vertex_counts) for a in args]
    new_counts = torch.ones(vertices.size(0), dtype=torch.long).fill_(3)
    return (vertices, new_counts, *new_attrs)


def import_mesh(file_path, scene_path=None, time=None):
    r"""Import a single mesh from a USD file in an unbatched representation.

    Supports homogeneous meshes (meshes with consistent numbers of vertices per face).
    All sub-meshes found under the `scene_path` are flattened to a single mesh. The following
    interpolation types are supported for UV coordinates: `vertex`, `varying` and `faceVarying`.
    Returns an unbatched representation.

    Args:
        file_path (str): Path to usd file (`\*.usd`, `\*.usda`).
        scene_path (str, optional): Scene path within the USD file indicating which primitive to import.
            If not specified, the all meshes in the scene will be imported and flattened into a single mesh.
        time (int, optional): Positive integer indicating the time at which to retrieve parameters.

    Returns:

    namedtuple of:
        - **vertices** (torch.FloatTensor): of shape (num_vertices, 3)
        - **faces** (torch.LongTensor): of shape (num_faces, face_size)
        - **uvs** (torch.FloatTensor): of shape (num_uvs, 2)
        - **face_uvs_idx** (torch.LongTensor): of shape (num_faces, face_size)
        - **face_normals** (torch.FloatTensor): of shape (num_faces, face_size, 3)
        - **materials** (list of kaolin.io.materials.Material): Material properties (Not yet implemented)

    Example:
        >>> # Create a stage with some meshes
        >>> stage = export_mesh('./new_stage.usd', vertices=torch.rand(3, 3), faces=torch.tensor([[0, 1, 2]]),
        ... scene_path='/World/mesh1')
        >>> # Import meshes
        >>> mesh = import_mesh(file_path='./new_stage.usd', scene_path='/World/mesh1')
        >>> mesh.vertices.shape
        torch.Size([3, 3])
        >>> mesh.faces
        tensor([[0, 1, 2]])
    """
    # TODO  add arguments to selectively import UVs, normals and materials
    if scene_path is None:
        scene_path = get_root(file_path)
    if time is None:
        time = Usd.TimeCode.Default()
    meshes_list = import_meshes(file_path, [scene_path], times=[time])
    return mesh_return_type(*meshes_list[0])


def import_meshes(file_path, scene_paths=None, heterogeneous_mesh_handler=None, times=None):
    r"""Import one or more meshes from a USD file in an unbatched representation.

    Supports homogeneous meshes (meshes with consistent numbers of vertices per face). Custom handling of
    heterogeneous meshes can be achieved by passing a function through the ``heterogeneous_mesh_handler`` argument.
    The following interpolation types are supported for UV coordinates: `vertex`, `varying` and `faceVarying`.
    For each scene path specified in `scene_paths`, sub-meshes (if any) are flattened to a single mesh.
    Returns unbatched meshes list representation. Prims with no meshes or with heterogenous faces are skipped.

    Args:
        file_path (str): Path to usd file (`\*.usd`, `\*.usda`).
        scene_paths (list of str, optional): Scene path(s) within the USD file indicating which primitive(s)
            to import. If None, all prims of type `Mesh` will be imported.
        heterogeneous_mesh_handler (function, optional): Optional function to handle heterogeneous meshes. The function's
            input and output must be  ``vertices`` (torch.FloatTensor), ``faces`` (torch.LongTensor),
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
        - **materials** (list of kaolin.io.materials.Material): Material properties (Not yet implemented)

    Example:
        >>> # Create a stage with some meshes
        >>> vertices_list = [torch.rand(3, 3) for _ in range(3)]
        >>> faces_list = [torch.tensor([[0, 1, 2]]) for _ in range(3)]
        >>> stage = export_meshes('./new_stage.usd', vertices=vertices_list, faces=faces_list)
        >>> # Import meshes
        >>> meshes = import_meshes(file_path='./new_stage.usd')
        >>> len(meshes)
        3
        >>> meshes[0].vertices.shape
        torch.Size([3, 3])
        >>> [m.faces for m in meshes]
        [tensor([[0, 1, 2]]), tensor([[0, 1, 2]]), tensor([[0, 1, 2]])]
    """
    # TODO  add arguments to selectively import UVs, normals and materials
    assert os.path.exists(file_path)
    stage = Usd.Stage.Open(file_path)
    if scene_paths is None:
        scene_paths = get_scene_paths(file_path, prim_types=['Mesh'])
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)

    vertices_list, faces_list, uvs_list, face_uvs_idx_list, face_normals_list, materials_list = [], [], [], [], [], []
    for scene_path, time in zip(scene_paths, times):
        mesh_attr = _get_flattened_mesh_attributes(stage, scene_path, time=time)
        vertices, face_vertex_counts, faces, uvs, face_uvs_idx, face_normals, materials = mesh_attr
        # TODO(jlafleche) Replace tuple output with mesh class

        if faces is not None:
            if not torch.all(face_vertex_counts == face_vertex_counts[0]):
                if heterogeneous_mesh_handler is None:
                    raise NonHomogeneousMeshError(f'Mesh at {scene_path} is non-homogeneous '
                                                  f'and cannot be imported from {file_path}.')
                else:
                    mesh = heterogeneous_mesh_handler(vertices, face_vertex_counts, faces, uvs, face_uvs_idx, face_normals)
                    if mesh is None:
                        continue
                    else:
                        vertices, face_vertex_counts, faces, uvs, face_uvs_idx, face_normals = mesh
            if faces.size(0) > 0:
                faces = faces.view(-1, face_vertex_counts[0])

        if face_uvs_idx is not None and faces is not None and face_uvs_idx.size(0) > 0:
            uvs = uvs.reshape(-1, 2)
            face_uvs_idx = face_uvs_idx.reshape(-1, faces.size(1))
        if face_normals is not None and faces is not None and face_normals.size(0) > 0:
            face_normals = face_normals.reshape(-1, faces.size(1), 3)

        vertices_list.append(vertices)
        faces_list.append(faces)
        uvs_list.append(uvs)
        face_uvs_idx_list.append(face_uvs_idx)
        face_normals_list.append(face_normals)
        materials_list.append(materials)

    params = [vertices_list, faces_list, uvs_list, face_uvs_idx_list, face_normals_list, materials_list]
    return [mesh_return_type(v, f, uv, fuv, fn, m) for v, f, uv, fuv, fn, m in zip(*params)]


def add_mesh(stage, scene_path, vertices=None, faces=None, uvs=None, face_uvs_idx=None, face_normals=None, time=None):
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
        face_normals (torch.Tensor, optional): of shape ``(num_vertices, num_faces, 3)``.
        time (int, optional): Positive integer defining the time at which the supplied parameters correspond to.
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
        faces_list = faces.view(-1).cpu().long().tolist()
        usd_mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts, time=time)
        usd_mesh.GetFaceVertexIndicesAttr().Set(faces_list, time=time)
    if vertices is not None:
        vertices_list = vertices.cpu().float().tolist()
        usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray(vertices_list), time=time)
    if uvs is not None:
        interpolation = None
        uvs_list = uvs.view(-1, 2).cpu().float().tolist()
        pv = UsdGeom.PrimvarsAPI(usd_mesh.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.Float2Array)
        pv.Set(uvs_list, time=time)
        if face_uvs_idx is not None:
            pv.SetIndices(Vt.IntArray(face_uvs_idx.view(-1).cpu().long().tolist()), time=time)
            interpolation = 'faceVarying'
        else:
            if vertices is not None and uvs.size(0) == vertices.size(0):
                interpolation = 'vertex'
            elif uvs.size(0) == faces.size(0):
                interpolation = 'uniform'
            elif uvs.size(0) == len(faces_list):
                interpolation = 'faceVarying'

        if interpolation is not None:
            pv.SetInterpolation(interpolation)
    if face_uvs_idx is not None and uvs is None:
        raise ValueError('If providing "face_uvs_idx", "uvs" must also be provided.')

    if face_normals is not None:
        face_normals = face_normals.view(-1, 3).cpu().float().tolist()
        usd_mesh.GetNormalsAttr().Set(face_normals, time=time)
        UsdGeom.PointBased(usd_mesh).SetNormalsInterpolation('faceVarying')

    return usd_mesh.GetPrim()


def export_mesh(file_path, scene_path='/World/Meshes/mesh_0', vertices=None, faces=None,
                uvs=None, face_uvs_idx=None, face_normals=None, materials=None, up_axis='Y', time=None):
    r"""Export a single mesh to USD.

    Export a single mesh defined by vertices and faces and save the stage to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_path (str, optional): Absolute path of mesh within the USD file scene. Must be a valid ``Sdf.Path``.
            If no path is provided, a default path is used.
        vertices (torch.FloatTensor, optional): Vertices with shape ``(num_vertices, 3)``.
        faces (torch.LongTensor, optional): Vertex indices for each face with shape ``(num_faces, face_size)``.
            Mesh must be homogenous (consistent number of vertices per face).
        uvs (torch.FloatTensor, optional): of shape ``(num_uvs, 2)``.
        face_uvs_idx (torch.LongTensor, optional): of shape ``(num_faces, face_size)``. If provided, `uvs` must also
            be specified.
        face_normals (torch.Tensor, optional): of shape ``(num_vertices, num_faces, 3)``.
        up_axis (str, optional): Specifies the scene's up axis. Choose from ``['Y', 'Z']``.
        time (int, optional): Positive integer defining the time at which the supplied parameters correspond to.
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
    else:
        stage = create_stage(file_path, up_axis)
    add_mesh(stage, scene_path, vertices, faces, uvs, face_uvs_idx, face_normals, time=time)
    stage.Save()

    return stage


def export_meshes(file_path, scene_paths=None, vertices=None, faces=None,
                  uvs=None, face_uvs_idx=None, face_normals=None, up_axis='Y', times=None):
    r"""Export multiple meshes to a new USD stage.

    Export multiple meshes defined by lists vertices and faces and save the stage to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_paths (list of str, optional): Absolute paths of meshes within the USD file scene. Must have the same number of 
            paths as the number of meshes ``N``. Must be a valid Sdf.Path. If no path is provided, a default path is used.
        vertices (list of torch.FloatTensor, optional): Vertices with shape ``(num_vertices, 3)``.
        faces (list of torch.LongTensor, optional): Vertex indices for each face with shape ``(num_faces, face_size)``.
            Mesh must be homogenous (consistent number of vertices per face).
        uvs (list of torch.FloatTensor, optional): of shape ``(num_uvs, 2)``.
        face_uvs_idx (list of torch.LongTensor, optional): of shape ``(num_faces, face_size)``. If provided, `uvs` must also
            be specified.
        face_normals (list of torch.Tensor, optional): of shape ``(num_vertices, num_faces, 3)``.
        up_axis (str, optional): Specifies the scene's up axis. Choose from ``['Y', 'Z']``.
        times (list of int, optional): Positive integers defining the time at which the supplied parameters correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> vertices_list = [torch.rand(3, 3) for _ in range(3)]
        >>> faces_list = [torch.tensor([[0, 1, 2]]) for _ in range(3)]
        >>> stage = export_meshes('./new_stage.usd', vertices=vertices_list, faces=faces_list)
    """
    stage = create_stage(file_path, up_axis)
    mesh_parameters = {'vertices': vertices, 'faces': faces, 'uvs': uvs,
                       'face_uvs_idx': face_uvs_idx, 'face_normals': face_normals}
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

    for i, scene_path in enumerate(scene_paths):
        mesh_params = {k: p[i] for k, p in supplied_parameters.items()}
        add_mesh(stage, scene_path, **mesh_params)
    stage.Save()

    return stage


# PointCloud Functions
def import_pointcloud(file_path, scene_path, time=None):
    r"""Import a single pointcloud from a USD file.

    Assumes that the USD pointcloud is interpreted using a point instancer. Converts the coordinates
    of each point instance to a point within the output pointcloud.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_path (str): Scene path within the USD file indicating which primitive to import.
        time (int, optional): Positive integer indicating the time at which to retrieve parameters.
    Returns:
        (torch.FloatTensor): Point coordinates.

    Example:
        >>> points = torch.rand(100, 3)
        >>> stage = export_pointcloud('./new_stage.usd', points, scene_path='/World/pointcloud')
        >>> points_imp = import_pointcloud(file_path='./new_stage.usd',
        ...                                scene_path='/World/pointcloud')
        >>> points_imp.shape
        torch.Size([100, 3])
    """
    if time is None:
        time = Usd.TimeCode.Default()
    pointcloud_list = import_pointclouds(file_path, [scene_path], times=[time])
    return pointcloud_list[0]


def import_pointclouds(file_path, scene_paths=None, times=None):
    r"""Import one or more pointclouds from a USD file.

    Assumes that pointclouds are interpreted using point instancers. Converts the coordinates
    of each point instance to a point within the output pointcloud.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_paths (list of str, optional): Scene path(s) within the USD file indicating which primitive(s)
            to import. If None, will return all pointclouds found based on PointInstancer prims with `kaolin_type`
            primvar set to `PointCloud`.
        times (list of int): Positive integers indicating the time at which to retrieve parameters.
    Returns:
        (list of torch.FloatTensor): Point coordinates.

    Example:
        >>> points = torch.rand(100, 3)
        >>> stage = export_pointclouds('./new_stage.usd', [points, points, points])
        >>> pointclouds = import_pointclouds(file_path='./new_stage.usd')
        >>> len(pointclouds)
        3
        >>> pointclouds[0].shape
        torch.Size([100, 3])
    """
    assert os.path.exists(file_path)
    stage = Usd.Stage.Open(file_path)

    # If scene path not specified, find all point clouds
    if scene_paths is None:
        scene_paths = []
        for p in stage.Traverse():
            is_point_instancer = UsdGeom.PointInstancer(p)
            if UsdGeom.PointInstancer(p) and p.GetAttribute('primvars:kaolin_type').Get() == 'PointCloud':
                scene_paths.append(p.GetPath())
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)

    pointclouds = []
    for scene_path, time in zip(scene_paths, times):
        prim = stage.GetPrimAtPath(scene_path)
        assert prim, f'The prim at {scene_path} does not exist.'

        instancer = UsdGeom.PointInstancer(prim)
        assert instancer   # Currently only support pointclouds from point instancers
        pointclouds.append(torch.tensor(instancer.GetPositionsAttr().Get(time=time)))
    return pointclouds


def add_pointcloud(stage, points, scene_path, time=None):
    r"""Add a pointcloud to an existing USD stage.

    Create a pointcloud represented by point instances of a sphere centered at each point coordinate.
    The stage is modified but not saved to disk.

    Args:
        stage (Usd.Stage): Stage onto which to add the pointcloud.
        points (torch.FloatTensor): Pointcloud tensor containing ``N`` points of shape ``(N, 3)``.
        scene_path (str): Absolute path of pointcloud within the USD file scene. Must be a valid Sdf.Path.
        time (int, optional): Positive integer defining the time at which the supplied parameters correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> stage = create_stage('./new_stage.usd')
        >>> points = torch.rand(100, 3)
        >>> stage = add_pointcloud(stage, points, '/World/PointClouds/pointcloud_0')
        >>> stage.Save()
    """
    scene_path = Sdf.Path(scene_path)
    if time is None:
        time = Usd.TimeCode.Default()

    if stage.GetPrimAtPath(scene_path):
        instancer_prim = stage.GetPrimAtPath(scene_path)
    else:
        instancer_prim = stage.DefinePrim(scene_path, 'PointInstancer')
    instancer = UsdGeom.PointInstancer(instancer_prim)
    assert instancer
    sphere = UsdGeom.Sphere.Define(stage, f'{scene_path}/sphere')
    sphere.GetRadiusAttr().Set(0.5)
    instancer.CreatePrototypesRel().SetTargets([sphere.GetPath()])

    # Calculate default point scale
    bounds = points.max(dim=0)[0] - points.min(dim=0)[0]
    min_bound = min(bounds)
    scale = (min_bound / points.size(0) ** (1 / 3)).item()

    # Generate instancer parameters
    indices = [0] * points.size(0)
    positions = points.cpu().tolist()
    scales = [(scale,) * 3] * points.size(0)

    # Populate point instancer
    instancer.GetProtoIndicesAttr().Set(indices, time=time)
    instancer.GetPositionsAttr().Set(positions, time=time)
    instancer.GetScalesAttr().Set(scales, time=time)

    # Create a primvar to identify the point instancer as a Kaolin PointCloud
    prim = stage.GetPrimAtPath(instancer.GetPath())
    pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar('kaolin_type', Sdf.ValueTypeNames.String)
    pv.Set('PointCloud')

    return stage


def export_pointcloud(file_path, pointcloud, scene_path='/World/PointClouds/pointcloud_0', time=None):
    r"""Export a single pointcloud to a USD scene.

    Export a single pointclouds to USD. The pointcloud will be added to the USD stage and represented
    by point instances of a sphere centered at each point coordinate. The stage is then saved to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        pointcloud (torch.FloatTensor): Pointcloud tensor containing ``N`` points of shape ``(N, 3)``.
        scene_path (str, optional): Absolute path of pointcloud within the USD file scene. Must be a valid Sdf.Path.
            If no path is provided, a default path is used.
        time (int): Positive integer defining the time at which the supplied parameters correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> points = torch.rand(100, 3)
        >>> stage = export_pointcloud('./new_stage.usd', points)
    """
    stage = export_pointclouds(file_path, [pointcloud], [scene_path], times=[time])
    return stage


def export_pointclouds(file_path, pointclouds, scene_paths=None, times=None):
    r"""Export one or more pointclouds to a USD scene.

    Export one or more pointclouds to USD. The pointclouds will be added to the USD stage and represented
    by point instances of a sphere centered at each point coordinate. The stage is then saved to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        pointclouds (list of torch.FloatTensor): List of pointcloud tensors containing ``N`` points of shape ``(N, 3)``.
        scene_paths (list of str, optional): Absolute path(s) of pointcloud(s) within the USD file scene. Must be a valid Sdf.Path.
            If no path is provided, a default path is used.
        times (list of int): Positive integers defining the time at which the supplied parameters correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> points = torch.rand(100, 3)
        >>> stage = export_pointcloud('./new_stage.usd', points)
    """
    if scene_paths is None:
        scene_paths = [f'/World/PointClouds/pointcloud_{i}' for i in range(len(pointclouds))]
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)

    assert len(pointclouds) == len(scene_paths)
    stage = create_stage(file_path)
    for scene_path, points, time in zip(scene_paths, pointclouds, times):
        add_pointcloud(stage, points, scene_path, time=time)
    stage.Save()

    return stage


# VoxelGrid Functions
def import_voxelgrid(file_path, scene_path, time=None):
    r"""Import a single voxelgrid from a USD file.

    Assumes that the USD voxelgrid is defined by a point instancer. Converts the coordinates
    of each point instance to an occupied voxel. The output grid size is determined by the `grid_size`
    primvar. If not specified, grid size will be determined by the axis with the largest number of occupied
    voxels. The output voxelgrid will be of shape ``[grid_size, grid_size, grid_size]``.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_path (str): Scene path within the USD file indicating which PointInstancer primitive
            to import as a voxelgrid.
        time (int, optional): Positive integer indicating the time at which to retrieve parameters.
    Returns:
        torch.BoolTensor

    Example:
        >>> voxelgrid = torch.rand(32, 32, 32) > 0.5
        >>> stage = export_voxelgrid('./new_stage.usd', voxelgrid, scene_path='/World/voxelgrid')
        >>> voxelgrid_imp = import_voxelgrid(file_path='./new_stage.usd',
        ...                                  scene_path='/World/voxelgrid')
        >>> voxelgrid_imp.shape
        torch.Size([32, 32, 32])
    """
    if time is None:
        time = Usd.TimeCode.Default()
    voxelgrid_list = import_voxelgrids(file_path, [scene_path], times=[time])
    return voxelgrid_list[0]

def import_voxelgrids(file_path, scene_paths=None, times=None):
    r"""Import one or more voxelgrids from a USD file.

    Assumes that the USD voxelgrid is defined by a point instancer. Converts the coordinates
    of each point instance to an occupied voxel. The output grid size is determined from the `grid_size`
    primvar. If not specified, grid size will be determined by the axis with the largest number of occupied
    voxels. The output voxelgrid will be of shape ``[grid_size, grid_size, grid_size]``.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_paths (list of str, optional): Scene path(s) within the USD file indicating which PointInstancer
            primitive(s) to import. If None, will return all pointclouds found based on PointInstancer
            prims with `kaolin_type` primvar set to `VoxelGrid`.
        times (list of int): Positive integers indicating the time at which to retrieve parameters.
    Returns:
        (list of torch.BoolTensor)

    Example:
        >>> voxelgrid_1 = torch.rand(32, 32, 32) > 0.5
        >>> voxelgrid_2 = torch.rand(32, 32, 32) > 0.5
        >>> stage = export_voxelgrids('./new_stage.usd', [voxelgrid_1, voxelgrid_2])
        >>> voxelgrid_imp = import_voxelgrids(file_path='./new_stage.usd')
        >>> len(voxelgrid_imp)
        2
        >>> voxelgrid_imp[0].shape
        torch.Size([32, 32, 32])
    """
    assert os.path.exists(file_path)
    stage = Usd.Stage.Open(file_path)

    # If scene path not specified, find all point clouds
    if scene_paths is None:
        scene_paths = []
        for p in stage.Traverse():
            is_point_instancer = UsdGeom.PointInstancer(p)
            if UsdGeom.PointInstancer(p) and p.GetAttribute('primvars:kaolin_type').Get() == 'VoxelGrid':
                scene_paths.append(p.GetPath())
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)

    voxelgrids = []
    for scene_path, time in zip(scene_paths, times):
        prim = stage.GetPrimAtPath(scene_path)
        assert prim, f'The prim at {scene_path} does not exist.'

        instancer = UsdGeom.PointInstancer(prim)
        assert instancer   # Currently only support pointclouds from point instancers

        voxel_indices = torch.tensor(instancer.GetPositionsAttr().Get(time=time), dtype=torch.long)
        bounds = voxel_indices.max(dim=0)[0]
        max_bound = bounds.max()
        grid_size = prim.GetAttribute('primvars:grid_size').Get(time=time)
        if grid_size is not None:
            assert max_bound < grid_size
        else:
            grid_size = max_bound
        voxelgrid = torch.zeros([grid_size, grid_size, grid_size], dtype=torch.bool)
        voxelgrid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.
        voxelgrids.append(voxelgrid)

    return voxelgrids


def add_voxelgrid(stage, voxelgrid, scene_path, time=None):
    r"""Add a voxelgrid to an existing USD stage.

    Add a voxelgrid where occupied voxels are defined by non-zero values. The voxelgrid is
    represented by point instances of a cube centered at each occupied index coordinate and scaled.
    The stage is modified but not saved to disk.

    Args:
        stage (Usd.Stage): Stage onto which to add the voxelgrid.
        voxelgrid (torch.BoolTensor): Binary voxelgrid of shape ``(N, N, N)``.
        scene_path (str): Absolute path of voxelgrid within the USD file scene. Must be a valid Sdf.Path.
        time (int, optional): Positive integer defining the time at which the supplied parameters correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> stage = create_stage('./new_stage.usd')
        >>> voxelgrid = torch.rand(32, 32, 32) > 0.5
        >>> stage = add_voxelgrid(stage, voxelgrid, '/World/VoxelGrids/voxelgrid_0')
        >>> stage.Save()
    """
    scene_path = Sdf.Path(scene_path)
    if time is None:
        time = Usd.TimeCode.Default()

    if stage.GetPrimAtPath(scene_path):
        instancer_prim = stage.GetPrimAtPath(scene_path)
    else:
        instancer_prim = stage.DefinePrim(scene_path, 'PointInstancer')
    instancer = UsdGeom.PointInstancer(instancer_prim)
    assert instancer
    cube = UsdGeom.Cube.Define(stage, f'{scene_path}/cube')
    cube.GetSizeAttr().Set(1.0)
    instancer.CreatePrototypesRel().SetTargets([cube.GetPath()])

    # Get each occupied voxel's centre coordinate
    points = torch.nonzero(voxelgrid, as_tuple=False).float()

    # Generate instancer parameters
    indices = [0] * points.shape[0]
    positions = points.cpu().tolist()
    scales = [(1.,) * 3] * points.size(0)

    # Populate point instancer
    instancer.GetProtoIndicesAttr().Set(indices, time=time)
    instancer.GetPositionsAttr().Set(positions, time=time)
    instancer.GetScalesAttr().Set(scales, time=time)

    # Centre at (0, 0, 0) and scale points
    adjusted_scale = 1. / (voxelgrid.size(0) - 1)
    scale_op = instancer.GetPrim().GetAttribute('xformOp:scale')
    if not scale_op:
        scale_op = UsdGeom.Xformable(instancer).AddScaleOp()
    scale_op.Set((adjusted_scale,) * 3, time=time)
    translate_op = instancer.GetPrim().GetAttribute('xformOp:translate')
    if not translate_op:
        UsdGeom.Xformable(instancer).AddTranslateOp()
    translate_op.Set((-0.5,) * 3, time=time)

    # Create a primvar to record the voxelgrid size
    prim = stage.GetPrimAtPath(instancer.GetPath())
    pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar('grid_size', Sdf.ValueTypeNames.Int)
    pv.Set(voxelgrid.size(0))

    # Create a primvar to identify the poitn instancer as a Kaolin VoxelGrid
    pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar('kaolin_type', Sdf.ValueTypeNames.String)
    pv.Set('VoxelGrid')

    return stage


def export_voxelgrid(file_path, voxelgrid, scene_path='/World/VoxelGrids/voxelgrid_0', time=None):
    r"""Export a single voxelgrid to a USD scene.

    Export a binary voxelgrid where occupied voxels are defined by non-zero values. The voxelgrid is
    represented by point instances of a cube centered at each occupied index coordinates. The voxelgrid
    will be scaled so that it fits within a unit cube. The stage is then saved to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        voxelgrid (torch.BoolTensor): Binary voxelgrid of shape ``(N, N, N)``.
        scene_path (str, optional): Absolute path of voxelgrid within the USD file scene. Must be a valid Sdf.Path.
            If no path is provided, a default path is used.
        time (int, optional): Positive integer defining the time at which the supplied parameters correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> voxelgrid = torch.rand(32, 32, 32) > 0.5
        >>> stage = export_voxelgrid('./new_stage.usd', voxelgrid)
    """
    if time is None:
        time = Usd.TimeCode.Default()
    stage = export_voxelgrids(file_path, [voxelgrid], [scene_path], times=[time])
    return stage


def export_voxelgrids(file_path, voxelgrids, scene_paths=None, times=None):
    r"""Export one or more voxelgrids to a USD scene.

    Export one or more binary voxelgrids where occupied voxels are defined by non-zero values. The voxelgrids are
    represented by point instances of a cube centered at each occupied index coordinates and scaled. The voxelgrids
    will be scaled so that it fits within a unit cube. The stage is then saved to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        voxelgrids (list of torch.BoolTensor): List of binary voxelgrid(s) of shape ``(N, N, N)``.
        scene_path (list of str, optional): Absolute path(s) of voxelgrid within the USD file scene. Must be a valid Sdf.Path.
            If no path is provided, a default path is used.
        times (list of int): Positive integers defining the time at which the supplied parameters correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> voxelgrid_1 = torch.rand(32, 32, 32) > 0.5
        >>> voxelgrid_2 = torch.rand(32, 32, 32) > 0.5
        >>> stage = export_voxelgrids('./new_stage.usd', [voxelgrid_1, voxelgrid_2])
    """
    if scene_paths is None:
        scene_paths = [f'/World/VoxelGrids/voxelgrid_{i}' for i in range(len(voxelgrids))]
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)
    assert len(voxelgrids) == len(scene_paths)
    stage = create_stage(file_path)
    for scene_path, voxelgrid, time in zip(scene_paths, voxelgrids, times):
        add_voxelgrid(stage, voxelgrid, scene_path, time=time)
    stage.Save()

    return stage
