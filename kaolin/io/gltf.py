# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pygltflib import GLTF2, Scene, ImageFormat, BufferFormat, \
                      SHORT, UNSIGNED_SHORT, UNSIGNED_INT, \
                      BYTE, UNSIGNED_BYTE, FLOAT, SCALAR, VEC2, VEC3, VEC4

import os
import copy
import warnings
import time
import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from .materials import PBRMaterial
from ..rep import SurfaceMesh

__all__ = [
    'import_mesh',
    'import_meshes',
]


enum2dtype = {
    BYTE: torch.int8,
    UNSIGNED_BYTE: torch.uint8,
    SHORT: torch.int16,
    UNSIGNED_SHORT: np.uint16,
    UNSIGNED_INT: torch.int32,
    FLOAT: torch.float
}

enum2lastdim = {
    SCALAR: 1,
    VEC2: 2,
    VEC3: 3,
    VEC4: 4
}

__alpha_modes = {'RGBA', 'LA'}
__none_alpha_modes = {'RGB', 'L'}
__supported_types = set.union(__alpha_modes, __none_alpha_modes)

def _make_rotation_mat(quat):
    x2 = quat[0] + quat[0]
    y2 = quat[1] + quat[1]
    z2 = quat[2] + quat[2]
    xx2 = quat[0] * x2
    xy2 = quat[0] * y2
    xz2 = quat[0] * z2
    yy2 = quat[1] * y2
    yz2 = quat[1] * z2
    zz2 = quat[2] * z2
    sx2 = quat[3] * x2
    sy2 = quat[3] * y2
    sz2 = quat[3] * z2
    return torch.tensor([
        [1 - (yy2 + zz2), xy2 + sz2, xz2 - sy2, 0],
        [xy2 - sz2, 1 - (xx2 + zz2), yz2 + sx2, 0],
        [xz2 + sy2, yz2 - sx2, 1 - (xx2 + yy2), 0],
        [0,         0,         0,               1]
    ], dtype=torch.double)

def _load_img(gltf, tex_idx, output_alpha):
    """Load images from index.

    Images are stored as raw binaries in the gltf
    that can be decoded with BytesIO + PIL.Image

    Arguments:
        gltf (pygltflib.GLTF2): the file to import from.
        tex_idx (int): The image index in the gltf.
        output_alpha (bool): if True, output the alpha channel

    Returns:
        (torch.Tensor or tuple(torch.Tensor, torch.Tensor)):

            - The image as tensor in HWC format
            - (optional): The alpha channel
    """
    binary_blob = gltf.binary_blob()
    img_idx = gltf.textures[tex_idx].source
    img_metadata = gltf.images[img_idx]
    if img_metadata.bufferView is not None:
        bufview = gltf.bufferViews[gltf.images[img_idx].bufferView]
        im_file = BytesIO(memoryview(binary_blob)[
            bufview.byteOffset:bufview.byteOffset + bufview.byteLength])
        img = Image.open(im_file)
    elif img_metadata.uri is not None:
        img = Image.open(os.path.join(
            gltf._path, img_metadata.uri))
    else:
        raise Exception(
            "something is missing to load the image:",
            gltf.images[img_idx]
        )

    if img.mode not in __supported_types:
        if output_alpha or img.mode == 'P':
            img = img.convert('RGBA')
        else:
            img = img.convert('RGB')
    img_t = torch.as_tensor(np.array(img))
    if output_alpha:
        if img.mode in __none_alpha_modes:
            alpha = torch.full((img.height, img.width, 1), 255, dtype=torch.uint8)
        else:
            alpha = img_t[..., -1:]

    if img.mode in __alpha_modes:
        img_t = img_t[..., :-1]
    elif img.mode == 'L':
        img_t = img_t.unsqueeze(-1)


    if output_alpha:
        return img_t, alpha
    else:
        return img_t

def _load_specular_workflow_material(gltf, mat):
    """Load specular workflow material properties from
    KHR_materials_pbrSpecularGlossiness
    """
    d = {'is_specular_workflow': True}
    if 'diffuseTexture' in mat:
        diffuse_texture = _load_img(gltf, mat['diffuseTexture']['index'], False)
        diffuse_texture = diffuse_texture * (torch.tensor(
            mat['diffuseFactor'])[:3].reshape(1, 1, 3) / 255.)
        d['diffuse_texture'] = diffuse_texture
    else:
        d['diffuse_color'] = torch.tensor(mat['diffuseFactor'][:3])
    if 'specularGlossinessTexture' in mat:
        specular_texture, glossiness_texture = _load_img(
            gltf, mat['specularGlossinessTexture']['index'], True)
        d['specular_texture'] = (specular_texture * (torch.tensor(
            mat['specularFactor']
        ).reshape(1, 1, 3) / 255.))
        d['roughness_texture'] = (1. - (glossiness_texture * (mat['glossinessFactor'] / 255.)))
    else:
        d['roughness_value'] = torch.tensor(1. - mat['glossinessFactor'])
        d['specular_color'] = torch.tensor(mat['specularFactor'])
    return d

def _load_metallic_workflow_material(gltf, mat):
    """Load metallic workflow material properties from
    pbrMetallicRoughness
    """
    d = {'is_specular_workflow': False}
    if mat.baseColorTexture is None:
        d['diffuse_color'] = torch.tensor(mat.baseColorFactor[:3])
    else:
        base_color_texture = _load_img(gltf, mat.baseColorTexture.index, False)
        base_color_texture = base_color_texture * (torch.tensor(
            mat.baseColorFactor
        )[:3].reshape(1, 1, 3) / 255.)
        d['diffuse_texture'] = base_color_texture

    if mat.metallicRoughnessTexture is None:
        if mat.roughnessFactor is None:
            d['roughness_value'] = torch.tensor(1.)
        else:
            d['roughness_value'] = torch.tensor(mat.roughnessFactor)
        if mat.metallicFactor is None:
            d['metallic_value'] = torch.tensor(1.)
        else:
            d['metallic_value'] = torch.tensor(mat.metallicFactor)
    else:
        metallic_roughness_texture = _load_img(
            gltf, mat.metallicRoughnessTexture.index, False)
        if metallic_roughness_texture.shape[-1] == 1:
            roughness_texture = metallic_roughness_texture
            metallic_texture = metallic_roughness_texture
        elif metallic_roughness_texture.shape[-1] == 3:
            roughness_texture = metallic_roughness_texture[..., 1:2]
            metallic_texture = metallic_roughness_texture[..., 2:3]
        d['roughness_texture'] = (
            roughness_texture * (mat.roughnessFactor / 255.)
        )
        d['metallic_texture'] = (
            metallic_texture * (mat.metallicFactor / 255.)
        )
    return d

def _get_materials(gltf):
    """get all materials from a pygltflib.GLTF2
    
    Arguments:
        gltf (pygltflib.GLTF2): the file to import from.

    Returns:
        (list of :class:`PBRMaterial`): The materials.
    """
    gltf.convert_images(ImageFormat.BUFFERVIEW)

    materials = []
    for mat in gltf.materials:
        d = {}
        # TODO(cfujitsang): add occlusion map
        # Prioritize the Kronos extension for specular-glossiness workflow
        # Some materials contains both metallic-roughness and specular-glossiness
        # but specular-glossiness can contain more information
        if 'KHR_materials_pbrSpecularGlossiness' in mat.extensions:
            d.update(_load_specular_workflow_material(
                gltf, mat.extensions['KHR_materials_pbrSpecularGlossiness']))
        elif mat.pbrMetallicRoughness is not None:
            d.update(_load_metallic_workflow_material(
                gltf, mat.pbrMetallicRoughness))
        if mat.normalTexture is not None:
            d['normals_texture'] = (
                _load_img(gltf, mat.normalTexture.index, False) * (2. / 255.) - 1.
            ) * torch.tensor(
                [mat.normalTexture.scale, mat.normalTexture.scale, 1.]
            ).reshape(1, 1, 3)
        pbr_mat = PBRMaterial(**d)
        materials.append(pbr_mat)
    return materials

def _get_tensor(gltf, idx):
    """Load tensor from index.

    Arguments:
        gltf (pygltflib.GLTF2): the file to import from.
        idx (int): The tensor index in the gltf.

    Returns:
        (torch.Tensor): The tensor
    """

    binary_blob = gltf.binary_blob()
    accessor = gltf.accessors[idx]
    if accessor.sparse is not None:
        raise TypeError("Cannot load sparse data yet")
    bufview = gltf.bufferViews[accessor.bufferView]
    dtype = enum2dtype[accessor.componentType]
    if dtype == np.uint16:
        sizeofdtype = 2
    elif dtype.is_floating_point:
        sizeofdtype = int(torch.finfo(dtype).bits / 8)
    else:
        sizeofdtype = int(torch.iinfo(dtype).bits / 8)
    lastdim = enum2lastdim[accessor.type]

    if bufview.byteStride is None or bufview.byteStride == lastdim * sizeofdtype:
        if dtype == np.uint16:
            output = torch.from_numpy(np.frombuffer(
                binary_blob,
                dtype=dtype,
                count=accessor.count * lastdim,
                offset=bufview.byteOffset + accessor.byteOffset
            ).astype(np.int32))
        else:
            output = torch.frombuffer(
                binary_blob,
                dtype=dtype,
                count=accessor.count * lastdim,
                offset=bufview.byteOffset + accessor.byteOffset
            ).reshape(-1, lastdim)
    else:
        raise ValueError("We currently don't support strided inputs")

    return output

def _join_meshes(meshes):
    """"""
    has_tangents = False
    has_uvs = False
    has_normals = False
    # We need to checks all the meshes first to presence of
    # tangents / uvs / normals to know if we want to
    # compute the missing ones
    for mesh in meshes:
        if mesh.has_attribute('vertex_tangents'):
            has_tangents = True
        if mesh.has_attribute('uvs'):
            has_uvs = True
        if mesh.has_attribute('normals'):
            has_normals = True

    cur_num_vertices = 0
    cur_num_uvs = 0
    faces = []
    vertices = []
    face_uvs_idx = [] if has_uvs else None
    uvs = [] if has_uvs else None
    tangents = [] if has_tangents else None
    normals = [] if has_normals else None
    material_assignments = []

    for mesh_idx, mesh in enumerate(meshes):
        faces.append(mesh.faces + cur_num_vertices)
        cur_num_vertices += mesh.vertices.shape[0]
        vertices.append(mesh.vertices)
        if has_uvs:
            _face_uvs_idx = mesh.face_uvs_idx
            if _face_uvs_idx is None:
                _face_uvs_idx = torch.full_like(mesh.faces, -1)
            face_uvs_idx.append(_face_uvs_idx + cur_num_uvs)
            _uvs = mesh.uvs
            if _uvs is None:
                _uvs = torch.empty((0, 2))
            cur_num_uvs += _uvs.shape[0]
            uvs.append(_uvs)
        if has_tangents:
            _tangents = mesh.vertex_tangents
            if _tangents is None:
                _tangents = torch.zeros_like(mesh.vertices)
            tangents.append(_tangents)
        if has_normals:
            normals.append(mesh.vertex_normals)
        material_assignments.append(mesh.material_assignments)
    faces = torch.cat(faces, dim=0)
    vertices = torch.cat(vertices, dim=0)
    if has_tangents:
        tangents = torch.cat(tangents, dim=0)
    if has_normals:
        normals = torch.cat(normals, dim=0)
    if has_uvs:
        face_uvs_idx = torch.cat(face_uvs_idx, dim=0)
        uvs = torch.cat(uvs, dim=0)
    material_assignments = torch.cat(material_assignments, dim=0)
    return SurfaceMesh(
        faces=faces, vertices=vertices, vertex_tangents=tangents,
        face_uvs_idx=face_uvs_idx, uvs=uvs, vertex_normals=normals,
        material_assignments=material_assignments
    )
   
def _get_meshes(gltf):
    """get all meshes from a pygltflib.GLTF2
    
    Arguments:
        gltf (pygltflib.GLTF2): the file to import from.

    Returns:
        (list of :class:`SurfaceMesh`): The meshes in the gltf file.
    """
    meshes = []
    for mesh_idx, mesh in enumerate(gltf.meshes):
        sub_meshes = []
        skip_mesh = True
        for j, primitive in enumerate(mesh.primitives):
            if primitive.mode != 4:
                warnings.warn(f"mode {primitive.mode} is currently not supported",
                              UserWarning)
                faces = torch.empty((0, 3), dtype=torch.long)
            else:
                skip_mesh = False
                faces = _get_tensor(gltf, primitive.indices).reshape(-1, 3).long()

            material_idx = primitive.material if primitive.material is not None else -1
            material_assignments = torch.full(
                (faces.shape[0],), material_idx, dtype=torch.short)
            vertices = _get_tensor(gltf, primitive.attributes.POSITION)
            if primitive.attributes.COLOR_0 is not None:
                warnings.warn(
                    "gltf loader don't support vertex color yet. " +
                    "Please make a github request if needed.",
                    UserWarning
                )
            if primitive.attributes.TANGENT is not None:
                tangents = _get_tensor(gltf, primitive.attributes.TANGENT)
                tangents = tangents[..., :3] * tangents[..., -1:]
            else:
                tangents = None
            if primitive.attributes.TEXCOORD_0 is not None:
                uvs = _get_tensor(gltf, primitive.attributes.TEXCOORD_0)
                face_uvs_idx = faces
            else:
                uvs = None
                face_uvs_idx = None
            if primitive.attributes.NORMAL is not None:
                normals = _get_tensor(gltf, primitive.attributes.NORMAL)
            else:
                normals = None
            if primitive.attributes.JOINTS_0 is not None:
                warnings.warn(
                    "gltf loader don't support vertex skinning yet. " +
                    "This mesh might appear in canonical pose. " +
                    "Please make a github request if needed.",
                    UserWarning
                )
            sub_meshes.append(SurfaceMesh(
                faces=faces, vertices=vertices, vertex_tangents=tangents,
                face_uvs_idx=face_uvs_idx, uvs=uvs, vertex_normals=normals,
                material_assignments=material_assignments
            ))
        if skip_mesh:
            meshes.append(None)
        else:
            meshes.append(sub_meshes)
    output = []
    for m_group in meshes:
        if m_group is None:
            output.append(None)
        else:
            output.append(_join_meshes(m_group))
    return output

def import_mesh(path):
    """Import mesh from a gltf (.glb or .gltf) file.

    Arguments:
        path (str): path to the gltf file.

    Returns:
        (kaolin.rep.SurfaceMesh): The imported mesh.
    """
    gltf = GLTF2.load(path)
    gltf.convert_buffers(BufferFormat.BINARYBLOB)
    materials = _get_materials(gltf)
    meshes = _get_meshes(gltf)
    for sampler in gltf.samplers:
        if sampler.wrapS != 10497 or sampler.wrapT != 10497:
            warnings.warn(
                "wrapping mode is not support yet. Please make a github request if needed.",
                UserWarning
            )
    default_scene = gltf.scenes[gltf.scene]
    scene_meshes = []

    has_tangents = False
    has_uvs = False
    has_normals = False
    for mesh in meshes:
        if mesh.has_attribute('vertex_tangents'):
            has_tangents = True
        if mesh.has_attribute('uvs'):
            has_uvs = True
        if mesh.has_attribute('normals'):
            has_normals = True

    def _traverse_scene(node_idx, cur_transform):
        node = gltf.nodes[node_idx]
        if node.matrix is not None:
            node_transform = torch.tensor(node.matrix, dtype=torch.double).reshape(4, 4)
        else:
            node_transform = None
            if node.scale is not None:
                node_transform = torch.tensor([
                    [node.scale[0], 0., 0., 0.],
                    [0., node.scale[1], 0., 0.],
                    [0., 0., node.scale[2], 0.],
                    [0., 0., 0., 1.]
                ], dtype=torch.double)
            if node.rotation is not None:
                rotation_mat = _make_rotation_mat(node.rotation)
                if node_transform is None:
                    node_transform = rotation_mat
                else:
                    node_transform = node_transform @ rotation_mat

            if node.translation is not None:
                translation_mat = torch.tensor([
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [node.translation[0], node.translation[1], node.translation[2], 1.]
                ], dtype=torch.double)
                if node_transform is None:
                    node_transform = translation_mat
                else:
                    node_transform = node_transform @ translation_mat
        if node_transform is not None:
            cur_transform = node_transform @ cur_transform
        if node.mesh is not None:
            mesh = copy.copy(meshes[node.mesh])
            if mesh is not None:
                vertices = torch.nn.functional.pad(
                    mesh.vertices, (0, 1), value=1., mode='constant')
                vertices = vertices @ cur_transform.float()
                mesh.vertices = vertices[..., :3]
                if has_tangents:
                    tangents = mesh.vertex_tangents @ cur_transform[:3, :3].float()
                    mesh.vertex_tangents = torch.nn.functional.normalize(
                        tangents, dim=-1)
                if has_normals:
                    inv_cur_transform = torch.linalg.inv(cur_transform[:3, :3]).T.float()
                    normals = mesh.vertex_normals @ inv_cur_transform
                    mesh.vertex_normals = torch.nn.functional.normalize(
                        normals, dim=-1)

                scene_meshes.append(mesh)
        for next_node_idx in node.children:
            _traverse_scene(next_node_idx, cur_transform)
    for node_idx in default_scene.nodes:
        _traverse_scene(node_idx, torch.eye(4, dtype=torch.double))
    outputs = _join_meshes(scene_meshes)
    outputs.materials = materials
    return outputs

def import_meshes(path):
    """Import meshes from a gltf (.glb or .gltf) file without them being composed in a scene.

    Arguments:
        path (str): path to the gltf file.

    Returns:
        (list of kaolin.rep.SurfaceMesh): The imported meshes.
    """
    gltf = GLTF2.load(path)
    gltf.convert_buffers(BufferFormat.BINARYBLOB)
    materials = _get_materials(gltf)
    meshes = _get_meshes(gltf)
    for m in meshes:
        global_assignments, local_assignments = torch.unique(m.material_assignments, return_inverse=True)
        m.materials = [materials[idx] for idx in global_assignments]
        m.material_assignments = local_assignments
    return meshes
