# Copyright (c) 2019, 20-21 NVIDIA CORPORATION. All rights reserved.
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
from pathlib import Path
import posixpath
from abc import abstractmethod
import warnings

import torch
from PIL import Image

try:
    from pxr import UsdShade, Sdf, Usd
except ImportError:
    warnings.warn("Warning: module pxr not found", ImportWarning)

from kaolin.io import usd


class Material:
    """Abstract material definition class.
    Defines material inputs and methods to export material properties.
    """
    @abstractmethod
    def write_to_usd(self, file_path, scene_path, bound_prims=None, time=None,
                     texture_dir=None, texture_file_prefix='', **kwargs):
        pass

    @abstractmethod
    def read_from_usd(self, file_path, scene_path, time=None):
        pass

    @abstractmethod
    def write_to_obj(self, obj_dir=None, texture_dir=None, texture_prefix=''):
        pass

    @abstractmethod
    def read_from_obj(self, file_path):
        pass


class PBRMaterial(Material):
    """Define a PBR material using USD Preview Surface.
    Usd Preview Surface (https://graphics.pixar.com/usd/docs/UsdPreviewSurface-Proposal.html)
    is a physically based surface material definition.

    Args:
        diffuse_color (tuple of floats): RGB values for `Diffuse` parameter (typically referred to as `Albedo`
            in a metallic workflow) in the range of `(0.0, 0.0, 0.0)` to `(1.0, 1.0, 1.0)`. Default value is grey
            `(0.5, 0.5, 0.5)`.
        roughness_value (float): Roughness value of specular lobe in range `0.0` to `1.0`. Default value is `0.5`.
        metallic_value (float): Typically set to `0.0` for non-metallic and `1.0` for metallic materials. Ignored
            if `is_specular_workflow` is `True`. Default value is `0.0`.
        specular_color (tuple of floats): RGB values for `Specular` lobe. Ignored if `is_specular_workflow` is
            `False`. Default value is white `(0.0, 0.0, 0.0)`.
        diffuse_texture (torch.FloatTensor): Texture for diffuse parameter, of shape `(3, height, width)`.
        roughness_texture (torch.FloatTensor): Texture for roughness parameter, of shape `(1, height, width)`.
        metallic_texture (torch.FloatTensor): Texture for metallic parameter, of shape `(1, height, width)`.
            Ignored if  `is_specular_workflow` is `True`.
        specular_texture (torch.FloatTensor): Texture for specular parameter, of shape `(3, height, width)`.
            Ignored if `is_specular_workflow` is `False`.
        normals_texture (torch.FloatTensor): Texture for normal mapping of shape `3, height, width)`.Normals
            maps create the illusion of fine three-dimensional detail without increasing the number of polygons.
            Tensor values must be in the range of `[(-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)]`.
        is_specular_workflow (bool): Determines whether or not to use a specular workflow. Default
            is `False` (use a metallic workflow).
    """

    def __init__(
        self,
        diffuse_color=(0.5, 0.5, 0.5),
        roughness_value=0.5,
        metallic_value=0.,
        specular_color=(0.0, 0.0, 0.0),
        diffuse_texture=None,
        roughness_texture=None,
        metallic_texture=None,
        specular_texture=None,
        normals_texture=None,
        displacement_texture=None,
        is_specular_workflow=False,
    ):
        self.diffuse_color = diffuse_color
        self.roughness_value = roughness_value
        self.metallic_value = metallic_value
        self.specular_color = specular_color
        self.diffuse_texture = diffuse_texture
        self.roughness_texture = roughness_texture
        self.metallic_texture = metallic_texture
        self.specular_texture = specular_texture
        self.normals_texture = normals_texture
        self.is_specular_workflow = is_specular_workflow

        self.shaders = {
            'UsdPreviewSurface': {
                'writer': self._write_usd_preview_surface,
                'reader': self._read_usd_preview_surface,
            },
        }

    def write_to_usd(self, file_path, scene_path, bound_prims=None, time=None,
                     texture_dir='', texture_file_prefix='', shader='UsdPreviewSurface'):
        r"""Write material to USD.
        Textures will be written to disk in the format 
        `{file_path}/{texture_dir}/{texture_file_prefix}{attr}.png` where `attr` is one of 
        [`diffuse`, `roughness`, `metallic`, `specular`, `normals`].

        Args:
            file_path (str): Path to usd file (\*.usd, \*.usda).
            scene_path (str): Absolute path of material within the USD file scene. Must be a valid ``Sdf.Path``.
            shader (str, optional): Name of shader to write. If not provided, use UsdPreviewSurface.
            bound_prims (list of Usd.Prim, optional): If provided, bind material to each prim.
            time (int, optional): Positive integer defining the time at which the supplied parameters correspond to.
            texture_dir (str, optional): Subdirectory to store texture files. If not provided, texture files will be
                saved in the same directory as the USD file specified by `file_path`.
            texture_file_prefix (str, optional): String to be prepended to the filename of each texture file.
        """
        assert os.path.splitext(file_path)[1] in ['.usd', '.usda'], f'Invalid file path "{file_path}".'
        assert shader in self.shaders, f'Shader {shader} is not support. Choose from {list(self.shaders.keys())}.'
        if os.path.exists(file_path):
            stage = Usd.Stage.Open(file_path)
        else:
            stage = usd.create_stage(file_path)
        if time is None:
            time = Usd.TimeCode.Default()

        writer = self.shaders[shader]['writer']
        return writer(stage, file_path, scene_path, bound_prims, time, texture_dir, texture_file_prefix)

    def _write_usd_preview_surface(self, stage, file_path, scene_path, bound_prims,
                                   time, texture_dir, texture_file_prefix):
        texture_dir = Path(texture_dir).as_posix()
        """Write a USD Preview Surface material."""
        material = UsdShade.Material.Define(stage, scene_path)

        shader = UsdShade.Shader.Define(stage, f'{scene_path}/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')

        # Create Inputs
        diffuse_input = shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f)
        roughness_input = shader.CreateInput('roughness', Sdf.ValueTypeNames.Float)
        specular_input = shader.CreateInput('specularColor', Sdf.ValueTypeNames.Color3f)
        metallic_input = shader.CreateInput('metallic', Sdf.ValueTypeNames.Float)
        normal_input = shader.CreateInput('normal', Sdf.ValueTypeNames.Normal3f)
        is_specular_workflow_input = shader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int)

        # Set constant values
        if self.diffuse_color is not None:
            diffuse_input.Set(tuple(self.diffuse_color), time=time)
        if self.roughness_value is not None:
            roughness_input.Set(self.roughness_value, time=time)
        if self.specular_color is not None:
            specular_input.Set(tuple(self.specular_color), time=time)
        if self.metallic_value is not None:
            metallic_input.Set(self.metallic_value, time=time)
        is_specular_workflow_input.Set(int(self.is_specular_workflow), time=time)

        # Export textures abd Connect textures to shader
        usd_dir = os.path.dirname(file_path)
        if self.diffuse_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f'{texture_file_prefix}diffuse.png')
            self._write_image(self.diffuse_texture, posixpath.join(usd_dir, rel_filepath))
            texture = self._add_texture_shader(
                stage, f'{scene_path}/diffuse_texture', rel_filepath, time=time, channels_out=3)
            inputTexture = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f)
            diffuse_input.ConnectToSource(inputTexture)
        if self.roughness_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f'{texture_file_prefix}roughness.png')
            self._write_image(self.roughness_texture, posixpath.join(usd_dir, rel_filepath))
            texture = self._add_texture_shader(
                stage, f'{scene_path}/roughness_texture', rel_filepath, time=time, channels_out=1)
            inputTexture = texture.CreateOutput("r", Sdf.ValueTypeNames.Float)
            roughness_input.ConnectToSource(inputTexture)
        if self.specular_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f'{texture_file_prefix}specular.png')
            self._write_image(self.specular_texture, posixpath.join(usd_dir, rel_filepath))
            texture = self._add_texture_shader(
                stage, f'{scene_path}/specular_texture', rel_filepath, time=time, channels_out=3)
            inputTexture = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f)
            specular_input.ConnectToSource(inputTexture)
        if self.metallic_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f'{texture_file_prefix}metallic.png')
            self._write_image(self.metallic_texture, posixpath.join(usd_dir, rel_filepath))
            texture = self._add_texture_shader(
                stage, f'{scene_path}/metallic_texture', rel_filepath, time=time, channels_out=1)
            inputTexture = texture.CreateOutput("r", Sdf.ValueTypeNames.Float)
            metallic_input.ConnectToSource(inputTexture)
        if self.normals_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f'{texture_file_prefix}normals.png')
            self._write_image(((self.normals_texture + 1.) / 2.), posixpath.join(usd_dir, rel_filepath))
            texture = self._add_texture_shader(
                stage, f'{scene_path}/normals_texture', rel_filepath, time=time, channels_out=3)
            inputTexture = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Normal3f)
            normal_input.ConnectToSource(inputTexture)

        # create Usd Preview Surface Shader outputs
        shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
        shader.CreateOutput('displacement', Sdf.ValueTypeNames.Token)

        # create material
        material.CreateSurfaceOutput().ConnectToSource(shader.GetOutput('surface'))
        material.CreateDisplacementOutput().ConnectToSource(shader.GetOutput('displacement'))

        # bind material to bound prims if provided
        if bound_prims is not None:
            for prim in bound_prims:
                binding_api = UsdShade.MaterialBindingAPI(prim)
                binding_api.Bind(material)
        stage.Save()
        return material

    def _add_texture_shader(self, stage, path, texture_path, time, channels_out=3, scale=None, bias=None):
        assert channels_out > 0 and channels_out <= 4
        texture = UsdShade.Shader.Define(stage, path)
        texture.CreateIdAttr('UsdUVTexture')
        texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texture_path, time=time)
        if scale is not None:
            texture.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(scale, time=time)
        if bias is not None:
            texture.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(bias, time=time)

        channels = ['r', 'b', 'g', 'a']
        for channel in channels[:channels_out]:
            texture.CreateOutput(channel, Sdf.ValueTypeNames.Float)

        return texture

    @staticmethod
    def _read_image(path):
        img = Image.open(str(path))
        img_tensor = ((torch.FloatTensor(img.getdata())).reshape(*img.size, -1) / 255.).permute(2, 0, 1)
        return img_tensor

    @staticmethod
    def _write_image(img_tensor, path):
        img_tensor_uint8 = (img_tensor * 255.).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8)
        img = Image.fromarray(img_tensor_uint8.squeeze().cpu().numpy())
        img.save(path)

    def read_from_usd(self, file_path, scene_path, texture_path=None, time=None):
        r"""Read material from USD.

        Args:
            file_path (str): Path to usd file (\*.usd, \*.usda).
            scene_path (str): Absolute path of UsdShade.Material prim within the USD file scene.
                Must be a valid ``Sdf.Path``.
            texture_path (str, optional): Path to textures directory. If the USD has absolute paths
                to textures, set to an empty string. By default, the textures will be assumed to be
                under the same directory as the USD specified by `file_path`.
            time (int, optional): Positive integer indicating the time at which to retrieve parameters.
        """
        if time is None:
            time = Usd.TimeCode.Default()
        if texture_path is None:
            texture_file_path = os.path.dirname(file_path)
        else:
            usd_dir = os.path.dirname(file_path)
            texture_file_path = posixpath.join(usd_dir, texture_file_path)
        stage = Usd.Stage.Open(file_path)
        material = UsdShade.Material(stage.GetPrimAtPath(scene_path))
        assert material

        surface_shader = material.GetSurfaceOutput().GetConnectedSource()[0]
        shader = UsdShade.Shader(surface_shader)
        if shader.GetImplementationSourceAttr().Get(time=time) == 'id':
            shader_name = UsdShade.Shader(surface_shader).GetShaderId()
        else:
            raise NotImplementedError
        inputs = surface_shader.GetInputs()

        reader = self.shaders[shader_name]['reader']
        return reader(inputs, texture_file_path, time)

    def _read_usd_preview_surface(self, inputs, texture_file_path, time):
        """Read UsdPreviewSurface material."""
        texture_file_path = Path(texture_file_path).as_posix()
        params = {}
        for i in inputs:
            name = i.GetBaseName()
            while i.HasConnectedSource():
                i = i.GetConnectedSource()[0].GetInputs()[0]
            value = i.Get(time=time)
            itype = i.GetTypeName()

            if 'diffuse' in name.lower() or 'albedo' in name.lower():
                if itype == Sdf.ValueTypeNames.Color3f:
                    self.diffuse_color = tuple(value)
                elif itype == Sdf.ValueTypeNames.Asset:
                    fp = posixpath.join(texture_file_path, value.path)
                    self.diffuse_texture = self._read_image(fp)
            elif 'roughness' in name.lower():
                if itype == Sdf.ValueTypeNames.Float:
                    self.roughness_value = value
                elif itype == Sdf.ValueTypeNames.Asset:
                    fp = posixpath.join(texture_file_path, value.path)
                    self.roughness_texture = self._read_image(fp)
            elif 'metallic' in name.lower():
                if itype == Sdf.ValueTypeNames.Float:
                    self.metallic_value = value
                elif itype == Sdf.ValueTypeNames.Asset:
                    fp = posixpath.join(texture_file_path, value.path)
                    self.metallic_texture = self._read_image(fp)
            elif 'specular' in name.lower():
                if itype == Sdf.ValueTypeNames.Color3f:
                    self.specular_color = tuple(value)
                elif itype == Sdf.ValueTypeNames.Asset:
                    fp = posixpath.join(texture_file_path, value.path)
                    self.specular_texture = self._read_image(fp)
                self.is_specular_workflow = True
            elif 'specular' in name.lower() and 'workflow' in name.lower():
                if itype == Sdf.ValueTypeNames.Bool:
                    self.is_specular_workflow = value
            elif 'normal' in name.lower():
                if itype == Sdf.ValueTypeNames.Asset:
                    fp = posixpath.join(texture_file_path, value.path)
                    self.normals_texture = self._read_image(fp) * 2. - 1.
        return self
