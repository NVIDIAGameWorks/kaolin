# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
from enum import Enum
import logging
import torch

import kaolin as kal
from kaolin.render.camera import Camera
from kaolin.render.lighting import SgLightingParameters
from kaolin.render.materials import PBRMaterial
from kaolin.rep import SurfaceMesh
from kaolin.render.mesh.nvdiffrast_context import nvdiffrast_is_available, default_nvdiffrast_context
from .common import default_lighting, default_material

if nvdiffrast_is_available():
    import nvdiffrast.torch


class RenderPass(str, Enum):
    # TODO: add normals relative to camera, mask, anti-aliasing, depth, wireframe
    face_idx = "face_idx"
    uvs = "uvs"
    albedo = "albedo"
    normals = "normals"
    roughness = "roughness"
    diffuse = "diffuse"
    specular = "specular"
    features = "features"
    render = "render"
    alpha = "alpha"


def render_mesh(camera: Camera, mesh: SurfaceMesh, lighting: SgLightingParameters = None,
                custom_materials=None, custom_material_assignments=None,
                backend=None, nvdiffrast_context=None):
    r"""Default easy-to-use differentiable rendering function. Is able to perform rasterization-based rendering
    with partial PBR material spec, if the input mesh comes with materials of type
    :class:`kaolin.render.materials.PBRMaterial`.

    Args:
        camera (Camera): single unbatched camera.
        mesh (SurfaceMesh): single unbatched mesh with bundled materials and material assignments.
        lighting (SgLightingParameters, optional): spherical Gaussians lighting parameters instance, if not set,
            will use default.
        custom_materials (list of PBRMaterial, optional): allows passing in materials distinct from mesh.materials
        custom_material_assignments (torch.LongTensor, optional): allows passing in material assignments distinct from
            mesh.material_assignments
        backend (str, optional): which backend to use for rasterization/interpolation. If not set, will use nvdiffrast
            if available in the environment, or else Kaolin CUDA rasterizer. Pass in "nvdiffrast" to force the use
            of nvdiffrast (or error if not installed), and pass in "cuda" to force Kaolin bundled CUDA implementation.
        nvdiffrast_context (optional): if using nvdiffrast, pass in optional context. If not set, default context will
            be created for device.

    Returns:
        (dict) from RenderPass enum value names to 1 x camera.height x camera.width x nchannels images,
            guaranteed to contain RenderPass.render.name key with a value. Output is not clamped.
    """
    # TODO: extend support to batches
    if len(camera) != 1:
        raise NotImplementedError(f'Render is only implemented for single unbatched camera.')
    if len(mesh) > 1:
        raise NotImplementedError(f'Render is only implemented for mesh of length 1, not {len(mesh)}.')
    mesh = mesh[0]  # make sure mesh is not batched
    materials = custom_materials or mesh.materials
    material_assignments = custom_material_assignments if custom_material_assignments is not None else mesh.material_assignments
    if lighting is None:
        lighting = default_lighting().to(camera.device)

    if backend is None:
        backend = "nvdiffrast" if nvdiffrast_is_available() else "cuda"

    if backend == "nvdiffrast":
        if nvdiffrast_context is None:
            nvdiffrast_context = default_nvdiffrast_context(device=camera.device, raise_error=True)
        rast_result = mesh_rasterize_interpolate_nvdiffrast(mesh, camera, nvdiffrast_context)
    elif backend == "cuda":
        rast_result = mesh_rasterize_interpolate_cuda(mesh, camera)
    else:
        raise ValueError(f'Unsupported backend {backend}, "nvdiffrast" and "cuda" are supported.')
    face_idx, im_base_normals, im_tangents, uv_map, im_features = rast_result

    if im_base_normals is not None:
        face_vertices_ndc = kal.ops.mesh.index_vertices_by_faces(
            camera.transform(mesh.vertices).unsqueeze(0), mesh.faces)
        face_normal_sign = kal.ops.mesh.face_normals(face_vertices_ndc)[..., 2]

        # TODO: assess if this actually works in all cases
        im_normal_sign = torch.sign(face_normal_sign[0, face_idx])
        im_normal_sign[face_idx == -1] = 0.
        im_base_normals *= im_normal_sign.unsqueeze(-1)

    albedo, spec_albedo, im_world_normals, im_roughness = texture_sample_materials(
        backend, face_idx, im_base_normals, materials,
        uv_map=uv_map, material_assignments=material_assignments, im_tangents=im_tangents)

    diffuse_img, specular_img, img = sg_shade(
        camera, face_idx, albedo, spec_albedo, im_roughness, im_world_normals,
        lighting.amplitude, lighting.direction, lighting.sharpness)

    res = {
        RenderPass.render.face_idx: face_idx,
        RenderPass.render.name: img,
        RenderPass.albedo.name: albedo,
        RenderPass.normals.name: im_world_normals,
        RenderPass.diffuse.name: diffuse_img,
        RenderPass.specular.name: specular_img,
        RenderPass.uvs.name: uv_map,
        RenderPass.features.name: im_features
    }
    res = {k: v for k, v in res.items() if v is not None}  # skip None values

    return res


def mesh_rasterize_interpolate_cuda(
        mesh, camera, normals_required=True, uvs_required=True, tangents_required=True, features_required=True):
    """ Performs rasterization and interpolation using bundled Kaolin CUDA kernel. Returns image-space values, given
    camera resolution, for attributes that are required and available, and `None` for others.

    Args:
        mesh (SurfaceMesh): unbatched surface mesh
        camera (Camera): single camera
        normals_required (bool): if True, will compute interpolated mesh normals, else return None
        uvs_required (bool): if True, and present in mesh, will compute interpolated mesh uvs, else return None
        tangents_required (bool): if True, will compute interpolated mesh tangents, else return None
        features_required (bool): if True, and present in mesh, will compute interpolated mesh features, else return None

    Returns:
        (tuple of): face_idx, im_normals, im_tangents, im_uvs, im_features
    """
    vertices_camera = camera.extrinsics.transform(mesh.vertices)
    vertices_image = camera.intrinsics.transform(vertices_camera)

    face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(vertices_camera, mesh.faces)
    face_vertices_image = kal.ops.mesh.index_vertices_by_faces(vertices_image, mesh.faces)[..., :2]

    in_face_features = []
    idx_normals = idx_uvs = idx_tangents = idx_features = -1
    current_idx = 0
    if normals_required:
        in_face_features.append(mesh.face_normals)
        idx_normals = current_idx
        current_idx += in_face_features[-1].shape[-1]
    if uvs_required and mesh.has_or_can_compute_attribute('face_uvs'):
        in_face_features.append(mesh.face_uvs)
        idx_uvs = current_idx
        current_idx += in_face_features[-1].shape[-1]
    if tangents_required and mesh.has_or_can_compute_attribute('face_tangents'):
        in_face_features.append(mesh.face_tangents)
        idx_tangents = current_idx
        current_idx += in_face_features[-1].shape[-1]
    if features_required and mesh.has_or_can_compute_attribute('face_features'):
        in_face_features.append(mesh.face_features)
        idx_features = current_idx
        current_idx += in_face_features[-1].shape[-1]

    if len(in_face_features) == 0:
        in_face_features = torch.zeros(tuple(list(mesh.faces.shape) + [1]), dtype=camera.dtype, device=camera.device)

    in_face_features = torch.cat(in_face_features, dim=-1).float()
    face_features, face_idx = kal.render.mesh.rasterize(
        camera.height, camera.width,
        face_features=in_face_features,
        face_vertices_z=face_vertices_camera[..., -1],  # can be face_vertices_image[..., -1] instead?
        face_vertices_image=face_vertices_image)

    im_normals = im_uvs = im_tangents = im_features = None
    if idx_normals >= 0:
        im_normals = face_features[..., idx_normals:idx_normals+3]
    if idx_uvs >= 0:
        im_uvs = face_features[..., idx_uvs:idx_uvs+2] % 1.
    if idx_tangents >= 0:
        im_tangents = face_features[..., idx_tangents:idx_tangents+3]
    if idx_features >= 0:
        im_features = face_features[..., idx_features:]

    return face_idx, im_normals, im_tangents, im_uvs, im_features


def mesh_rasterize_interpolate_nvdiffrast(
        mesh, camera, nvdiffrast_context,
        normals_required=True, uvs_required=True, tangents_required=True, features_required=True):
    """ Performs rasterization and interpolation using nvdiffrast. Returns image-space values, given
        camera resolution, for attributes that are required and available, and `None` for others.

        Args:
            mesh (SurfaceMesh): unbatched surface mesh
            nvdiffrast_context (render context): nvdiffrast context, either CUDA or OpenGL
            camera (Camera): single camera
            normals_required (bool): if True, will compute interpolated mesh normals, else return None
            uvs_required (bool): if True, and present in mesh, will compute interpolated mesh uvs, else return None
            tangents_required (bool): if True, will compute interpolated mesh tangents, else return None
            features_required (bool): if True, and present in mesh, will compute interpolated mesh features, else return None

        Returns:
            (tuple of): face_idx, im_normals, im_tangents, im_uvs, im_features
        """
    vertices_camera = camera.extrinsics.transform(mesh.vertices)
    vertices_clip = camera.intrinsics.project(vertices_camera)

    faces_int = mesh.faces.int()
    rast = nvdiffrast.torch.rasterize(nvdiffrast_context, vertices_clip, faces_int,
                                      (camera.height, camera.width), grad_db=False)
    # Nvdiffrast rasterization contains u, v, z/w, triangle_id onf shape 1 x W x H x 4
    rast0 = torch.flip(rast[0], dims=(1,))  # why filp?
    face_idx = (rast0[..., -1].long() - 1).contiguous()

    im_normals = im_uvs = im_tangents = im_features = None
    if normals_required:
        # TODO: this implies smooth shading; can we do flat shading too?
        if mesh.has_or_can_compute_attribute('normals') and mesh.has_or_can_compute_attribute('face_normals_idx'):
            im_normals = nvdiffrast.torch.interpolate(mesh.normals, rast0, mesh.face_normals_idx.int())[0]
        elif mesh.has_attribute('face_normals'):
            # E.g. when importing from USD, we have normals for each vertex of a face, but not corresponding index
            normals, face_normals_idx = mesh.ensure_indexed_attribute('normals')
            if normals is not None and face_normals_idx is not None:
                im_normals = nvdiffrast.torch.interpolate(normals, rast0, face_normals_idx.int())[0]
        elif mesh.has_or_can_compute_attribute('vertex_normals'):  # should always be true
            im_normals = nvdiffrast.torch.interpolate(
                mesh.vertex_normals, rast0, faces_int)[0]

    if uvs_required:
        if mesh.has_or_can_compute_attribute('uvs') and mesh.has_or_can_compute_attribute('face_uvs_idx'):
            im_uvs = nvdiffrast.torch.interpolate(
                mesh.uvs, rast0, mesh.face_uvs_idx.int())[0] % 1.

    if tangents_required:
        if mesh.has_or_can_compute_attribute('vertex_tangents'):
            im_tangents = nvdiffrast.torch.interpolate(
                mesh.vertex_tangents, rast0, faces_int)[0]

    if features_required:
        if mesh.has_or_can_compute_attribute('face_features'):
            val, idx = kal.ops.mesh.unindex_vertices_by_faces(mesh.face_features)
            im_features = nvdiffrast.torch.interpolate(
                val, rast0, idx.int())[0]

    return face_idx, im_normals, im_tangents, im_uvs, im_features


def texture_sample_materials(backend, face_idx, im_base_normals, materials=None, uv_map=None,
                             material_assignments=None, im_tangents=None):
    """ Perform texture sampling on all material channel images.
    """
    height = face_idx.shape[-2]
    width = face_idx.shape[-1]
    device = face_idx.device

    albedo = torch.zeros((1, height, width, 3), device=device)
    spec_albedo = torch.zeros((1, height, width, 3), device=device)
    im_world_normals = torch.zeros((1, height, width, 3), device=device)
    im_roughness = torch.zeros((1, height, width, 1), device=device) + 0.5

    if materials is None:
        materials = [default_material().to(face_idx.device)]

    # Image-space material assignments
    if material_assignments is not None:
        im_material_idx = material_assignments[face_idx]
    else:  # assume first material for all
        im_material_idx = torch.zeros((1, height, width), dtype=torch.int16, device=device)
    im_material_idx[face_idx == -1] = -1

    im_bitangents = None
    if im_tangents is not None and im_base_normals is not None:
        im_bitangents = torch.nn.functional.normalize(torch.cross(im_tangents, im_base_normals), dim=-1)

    for i, material in enumerate(materials):
        mask = im_material_idx == i
        if mask.count_nonzero().item() == 0:  # material is not visible
            continue

        perturbation_normal = mapped_albedo = mapped_spec = mapped_metalic = mapped_roughness = None
        if uv_map is None:
            logging.warning(f'Missing uvmap; cannot texturemap materials')
        else:
            texcoords = uv_map[mask].reshape(1, 1, -1, 2).contiguous()
            if backend == 'nvdiffrast':
                interp_res = _texture_sample_material_nvdiffrast(material.hwc(), texcoords)
            elif backend == 'cuda':
                texcoords[..., 1] = 1 - texcoords[..., 1]
                interp_res = _texture_sample_material_cuda(material.chw(), texcoords)
            else:
                raise ValueError(f'Unsupported backend {backend}, "nvdiffrast" and "cuda" are supported.')
            perturbation_normal, mapped_albedo, mapped_spec, mapped_metalic, mapped_roughness = interp_res

        if perturbation_normal is not None and im_tangents is not None and im_bitangents is not None:
            perturbation_normal = perturbation_normal.unsqueeze(0).unsqueeze(0)
            shading_normals = torch.nn.functional.normalize(
                im_tangents[mask] * perturbation_normal[..., :1]
                - im_bitangents[mask] * perturbation_normal[..., 1:2]
                + im_base_normals[mask] * perturbation_normal[..., 2:3],
                dim=-1
            )
            im_world_normals[mask] = shading_normals
        else:
            im_world_normals[mask] = im_base_normals[mask]

        if mapped_albedo is not None:
            albedo[mask] = mapped_albedo[..., :3]  # TODO: handle opacity (not trivial)
        elif material.diffuse_color is not None:
            albedo[mask] = material.diffuse_color.unsqueeze(0)

        if material.is_specular_workflow:
            if mapped_spec is not None:
                spec_albedo[mask] = mapped_spec
            elif material.specular_color is not None:
                spec_albedo[mask] = material.specular_color.unsqueeze(0)
        else:
            metalic_val = mapped_metalic if mapped_metalic is not None else material.metallic_value
            if metalic_val is not None:
                # https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metal-brdf-and-dielectric-brdf
                spec_albedo[mask] = (1. - metalic_val) * 0.04 + albedo[mask] * metalic_val
                albedo[mask] = albedo[mask] * (1. - metalic_val)

        min_roughness = 1e-3
        if mapped_roughness is not None:
            im_roughness[mask] = mapped_roughness.clamp(min=min_roughness)
        elif material.roughness_value is not None:
            im_roughness[mask] = material.roughness_value.unsqueeze(0).clamp(min=min_roughness)

    return albedo, spec_albedo, im_world_normals, im_roughness


def _texture_sample_material_nvdiffrast(material, texcoords):
    def _proc_channel(texture_image):
        if texture_image is None:
            return None
        return nvdiffrast.torch.texture(
            texture_image.unsqueeze(0), texcoords, filter_mode='linear')[0, 0]  # first 2 dims are 1, 1

    mapped_normal = _proc_channel(material.normals_texture)
    mapped_albedo = _proc_channel(material.diffuse_texture)
    mapped_spec = _proc_channel(material.specular_texture)
    mapped_metalic = _proc_channel(material.metallic_texture)
    mapped_roughness = _proc_channel(material.roughness_texture)

    return mapped_normal, mapped_albedo, mapped_spec, mapped_metalic, mapped_roughness


def _texture_sample_material_cuda(material, texcoords):
    texture_images = []
    current_idx = 0

    def _texture_map(_texcoords, _image):
        return kal.render.mesh.texture_mapping(
            _texcoords, _image.unsqueeze(0), mode='bilinear').squeeze(0).squeeze(0)

    # Concatenate all maps into one image, if possible
    name_idx_nchannels = {}
    for name, image in zip(
            ['normals', 'diffuse', 'specular',
             'metalic', 'roughness'],
            [material.normals_texture, material.diffuse_texture, material.specular_texture,
             material.metallic_texture, material.roughness_texture]):
        if image is not None:
            compatible = len(texture_images) == 0 or texture_images[0].shape[-2:] == image.shape[-2:]
            if compatible:  # Can concatenate with other texture images (common case)
                texture_images.append(image)
                nchannels = image.shape[-3]
                name_idx_nchannels[name] = (current_idx, nchannels)
                current_idx += nchannels
            else:  # Else, we'll texture map it separately
                name_idx_nchannels[name] = _texture_map(texcoords, image)
        else:
            name_idx_nchannels[name] = (-1, 0)

    if len(texture_images) > 0:
        # Jointly process the textures that could concatenate
        texture_images = torch.cat(texture_images, dim=-3).float()
        rendered_images = _texture_map(texcoords, texture_images)

    def _proc_attr(name):
        preproc = name_idx_nchannels[name]
        if torch.is_tensor(preproc):  # Is tensor if not concatenatable and was processed separately
            return preproc
        idx, nchannels = preproc
        if idx >= 0:
            return rendered_images[..., idx:idx+nchannels]
        return None

    mapped_normal = _proc_attr('normals')
    mapped_albedo = _proc_attr('diffuse')
    mapped_spec = _proc_attr('specular')
    mapped_metalic = _proc_attr('metalic')
    mapped_roughness = _proc_attr('roughness')
    return mapped_normal, mapped_albedo, mapped_spec, mapped_metalic, mapped_roughness


def sg_shade(camera, face_idx, albedo, spec_albedo, im_roughness, im_world_normals, amplitude, direction, sharpness):
    """ Performs partial PBR shading of the materials, given SG lighting parameters and image space maps.
    """
    height = face_idx.shape[-2]
    width = face_idx.shape[-1]
    device = face_idx.device

    hard_mask = face_idx >= 0
    img = torch.zeros((1, height, width, 3), dtype=torch.float, device=device)

    _im_world_normals = torch.nn.functional.normalize(
        im_world_normals[hard_mask], dim=-1)
    diffuse_effect = kal.render.lighting.sg_diffuse_inner_product(
        amplitude, direction, sharpness,
        _im_world_normals,
        albedo[hard_mask]
    )
    img[hard_mask] = diffuse_effect
    diffuse_img = torch.zeros_like(img)
    diffuse_img[hard_mask] = diffuse_effect

    pixel_grid = kal.render.camera.raygen.generate_centered_custom_resolution_pixel_coords(
        img_width=camera.width, img_height=camera.height, res_x=width, res_y=height, device=camera.device
    )
    _, rays_d = kal.render.camera.raygen.generate_pinhole_rays(camera, pixel_grid)
    rays_d = rays_d.reshape(1, height, width, 3)
    specular_effect = kal.render.lighting.sg_warp_specular_term(
        amplitude, direction, sharpness,
        _im_world_normals,
        im_roughness[hard_mask].squeeze(-1),
        -rays_d[hard_mask],
        spec_albedo[hard_mask]
    )
    img[hard_mask] += specular_effect
    specular_img = torch.zeros_like(img)
    specular_img[hard_mask] = specular_effect

    return diffuse_img, specular_img, img
