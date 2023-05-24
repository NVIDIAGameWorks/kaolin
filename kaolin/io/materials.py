# Copyright (c) 2019,20-21 NVIDIA CORPORATION & AFFILIATES.
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

from abc import abstractmethod
from collections.abc import Callable, Mapping
import inspect
import os
from pathlib import Path
from PIL import Image
import posixpath
import torch
import warnings


try:
    from pxr import UsdShade, Sdf, Usd
except ImportError:
    warnings.warn('Warning: module pxr not found', ImportWarning)

from .usd.utils import create_stage


class MaterialError(Exception):
    pass


class MaterialNotSupportedError(MaterialError):
    pass


class MaterialLoadError(MaterialError):
    pass


class MaterialWriteError(MaterialError):
    pass


class MaterialFileError(MaterialError):
    pass


class MaterialNotFoundError(MaterialError):
    pass


def _get_shader_parameters(shader, time):
    # Get shader parameters
    params = {}
    inputs = shader.GetInputs()
    for i in inputs:
        name = i.GetBaseName()
        params.setdefault(i.GetBaseName(), {})
        if UsdShade.ConnectableAPI.HasConnectedSource(i):
            connected_source = UsdShade.ConnectableAPI.GetConnectedSource(i)
            connected_inputs = connected_source[0].GetInputs()
            while connected_inputs:
                connected_input = connected_inputs.pop()
                if UsdShade.ConnectableAPI.HasConnectedSource(connected_input):
                    new_inputs = UsdShade.ConnectableAPI.GetConnectedSource(connected_input)[0].GetInputs()
                    connected_inputs.extend(new_inputs)
                elif connected_input.Get(time=time) is not None:
                    params[name].setdefault(connected_input.GetBaseName(), {}).update({
                        'value': connected_input.Get(time=time),
                        'type': connected_input.GetTypeName().type,
                        'docs': connected_input.GetDocumentation(),
                    })
        else:
            params[name].update({
                'value': i.Get(time=time),
                'type': i.GetTypeName().type,
                'docs': i.GetDocumentation(),
            })
    return params


class MaterialManager:
    """Material management utility.
    Allows material reader functions to be mapped against specific shaders. This allows USD import functions
    to determine if a reader is available, which material reader to use and which material representation to wrap the
    output with.

    Default registered readers:

    - UsdPreviewSurface: Import material with shader id `UsdPreviewSurface`. All parameters are supported,
      including textures. See https://graphics.pixar.com/usd/release/wp_usdpreviewsurface.html for more details
      on available material parameters.

    Example:
        >>> # Register a new USD reader for mdl `MyCustomPBR`
        >>> from kaolin.io import materials
        >>> dummy_reader = lambda params, texture_path, time: UsdShade.Material()
        >>> materials.MaterialManager.register_usd_reader('MyCustomPBR', dummy_reader)
    """
    _usd_readers = {}
    _obj_reader = None

    @classmethod
    def register_usd_reader(cls, shader_name, reader_fn):
        """Register a shader reader function that will be used during USD material import.

        Args:
            shader_name (str): Name of the shader
            reader_fn (Callable): Function that will be called to read shader attributes. The function must take as
                input a dictionary of input parameters, a string representing the texture path, and a time
                `(params, texture_path, time)` and typically return a `Material`
        """
        if shader_name in cls._usd_readers:
            warnings.warn(f'Shader {shader_name} is already registered. Overwriting previous definition.')

        if not isinstance(reader_fn, Callable):
            raise MaterialLoadError('The supplied `reader_fn` must be a callable function.')

        # Validate reader_fn expects 3 parameters
        if len(inspect.signature(reader_fn).parameters) != 3:
            raise ValueError('Error encountered when validating supplied `reader_fn`. Ensure that '
                             'the function takes 3 arguments: parameters (dict), texture_path (string) and time '
                             '(float)')

        cls._usd_readers[shader_name] = reader_fn

    @classmethod
    def read_from_file(cls, file_path, scene_path=None, texture_path=None, time=None):
        r"""Read USD material and return a Material object.
        The shader used must have a corresponding registered reader function.

        Args:
            file_path (str): Path to usd file (\*.usd, \*.usda).
            scene_path (str): Required only for reading USD files. Absolute path of UsdShade.Material prim
                within the USD file scene. Must be a valid ``Sdf.Path``.
            texture_path (str, optional): Path to textures directory. By default, the textures will be assumed to be
                under the same directory as the file specified by `file_path`.
            time (convertible to float, optional): Optional for reading USD files. Positive integer indicating the tim
                at which to retrieve parameters.

        Returns:
            (Material): Material object determined by the corresponding reader function.
        """
        if os.path.splitext(file_path)[1] in ['.usd', '.usda', '.usdc']:
            if scene_path is None or not Sdf.Path(scene_path):
                raise MaterialLoadError(f'The scene_path `{scene_path}`` provided is invalid.')

            if texture_path is None:
                texture_file_path = os.path.dirname(file_path)
            elif not os.path.isabs(texture_path):
                usd_dir = os.path.dirname(file_path)
                texture_file_path = posixpath.join(usd_dir, texture_path)
            else:
                texture_file_path = texture_path

            stage = Usd.Stage.Open(file_path)
            material = UsdShade.Material(stage.GetPrimAtPath(scene_path))

            return cls.read_usd_material(material, texture_path=texture_file_path, time=time)

        elif os.path.splitext(file_path)[1] == '.obj':
            if cls._obj_reader is not None:
                return cls._obj_reader(file_path)
            else:
                raise MaterialNotSupportedError('No registered .obj material reader found.')

    @classmethod
    def read_usd_material(cls, material, texture_path, time=None):
        r"""Read USD material and return a Material object.
        The shader used must have a corresponding registered reader function. If no available reader is found,
        the material parameters will be returned as a dictionary.

        Args:
            material (UsdShade.Material): Valid USD Material prim
            texture_path (str, optional): Path to textures directory. If the USD has absolute paths
                to textures, set to an empty string. By default, the textures will be assumed to be
                under the same directory as the USD specified by `file_path`.
            time (convertible to float, optional): Positive integer indicating the time at which to retrieve parameters.

        Returns:
            (Material): Material object determined by the corresponding reader function.
        """
        if time is None:
            time = Usd.TimeCode.Default()

        if not UsdShade.Material(material):
            raise MaterialLoadError(f'The material `{material}` is not a valid UsdShade.Material object.')

        for surface_output in material.GetSurfaceOutputs():
            if not surface_output.HasConnectedSource():
                continue
            surface_shader = surface_output.GetConnectedSource()[0]
            shader = UsdShade.Shader(surface_shader)
            if not UsdShade.Shader(shader):
                raise MaterialLoadError(f'The shader `{shader}` is not a valid UsdShade.Shader object.')

            if shader.GetImplementationSourceAttr().Get(time=time) == 'id':
                shader_name = UsdShade.Shader(surface_shader).GetShaderId()
            elif shader.GetPrim().HasAttribute('info:mdl:sourceAsset'):
                # source_asset = shader.GetPrim().GetAttribute('info:mdl:sourceAsset').Get(time=time)
                shader_name = shader.GetPrim().GetAttribute('info:mdl:sourceAsset:subIdentifier').Get(time=time)
            else:
                shader_name = ''
                warnings.warn(f'A reader for the material defined by `{material}` is not yet implemented.')

            params = _get_shader_parameters(surface_shader, time)

            if shader_name not in cls._usd_readers:
                warnings.warn('No registered readers were able to process the material '
                              f'`{material}` with shader `{shader_name}`.')
                return params

            reader = cls._usd_readers[shader_name]
            return reader(params, texture_path, time)
        raise MaterialError(f'Error processing material {material}')


class Material:
    """Abstract material definition class.
    Defines material inputs and methods to export material properties.
    """
    def __init__(self, name):
        self.material_name = name

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
    """Define a PBR material
    Supports USD Preview Surface (https://graphics.pixar.com/usd/docs/UsdPreviewSurface-Proposal.html),
    a physically based surface material definition.

    Parameters:
        diffuse_color (tuple of floats):
            RGB values for `Diffuse` parameter (typically referred to as `Albedo`
            in a metallic workflow) in the range of `(0.0, 0.0, 0.0)` to `(1.0, 1.0, 1.0)`.
            Default value is grey `(0.5, 0.5, 0.5)`.
        roughness_value (float):
            Roughness value of specular lobe in range `0.0` to `1.0`. Default value is `0.5`.
        metallic_value (float):
            Metallic value, typically set to `0.0` for non-metallic and `1.0` for metallic materials. 
            Ignored if `is_specular_workflow` is `True`. Default value is `0.0`.
        clearcoat_value (float):
            Second specular lobe amount. Color is hardcoded to white. Default value is `0.0`.
        clearcoat_roughness_value (float):
            Roughness for the clearcoat specular lobe in the range `0.0` to `1.0`.
            The default value is `0.01`.
        opacity_value (float):
            Opacity, with `1.0` fully opaque and `0.0` as transparent with values within this range
            defining a translucent surface. Default value is `1.0`.
        opacity_treshold (float):
            Used to create cutouts based on the `opacity_value`. Surfaces with an opacity
            smaller than the `opacity_threshold` will be fully transparent. Default value is `0.0`.
        ior_value (float):
            Index of Refraction used with translucent objects and objects with specular components.
            Default value is `1.5`.
        specular_color (tuple of floats):
            RGB values for `Specular` lobe. Ignored if `is_specular_workflow` is `False`.
            Default value is white `(0.0, 0.0, 0.0)`.
        displacement_value (float):
            Displacement in the direction of the normal. Default is `0.0`
        diffuse_texture (torch.FloatTensor):
            Texture for diffuse parameter, of shape `(3, height, width)`.
        roughness_texture (torch.FloatTensor):
            Texture for roughness parameter, of shape `(1, height, width)`.
        metallic_texture (torch.FloatTensor):
            Texture for metallic parameter, of shape `(1, height, width)`.
            Ignored if  `is_specular_workflow` is `True`.
        clearcoat_texture (torch.FloatTensor):
            Texture for clearcoat parameter, of shape `(1, height, width)`.
        clearcoat_roughness_texture (torch.FloatTensor):
            Texture for clearcoat_roughness parameter, of shape
            `(1, height, width)`.
        opacity_texture (torch.FloatTensor):
            Texture for opacity parameter, of shape `(1, height, width)`.
        ior_texture (torch.FloatTensor):
            Texture for opacity parameter, of shape `(1, height, width)`.
        specular_texture (torch.FloatTensor):
            Texture for specular parameter, of shape `(3, height, width)`.
            Ignored if `is_specular_workflow` is `False`.
        normals_texture (torch.FloatTensor):
            Texture for normal mapping of shape `(3, height, width)`.
            Normals maps create the illusion of fine three-dimensional
            detail without increasing the number of polygons.
            Tensor values must be in the range of `[(-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)]`.
        displacement_texture (torch.FloatTensor):
            Texture for displacement in the direction of the normal `(1, height, width)`.
        is_specular_workflow (bool):
            Determines whether or not to use a specular workflow.
            Default is `False` (use a metallic workflow).
        diffuse_colorspace (string):
            Colorspace of texture, if provided. Select from [auto, raw, sRGB].
            Default is 'auto'.
        roughness_colorspace (string):
            Colorspace of texture, if provided. Select from [auto, raw].
            Default is 'auto'.
        metallic_colorspace (string):
            Colorspace of texture, if provided. Select from [auto, raw].
            Default is 'auto'.
        clearcoat_colorspace (string):
            Colorspace of texture, if provided. Select from [auto, raw].
            Default is 'auto'.
        clearcoat_roughness_colorspace (string):
            Colorspace of texture, if provided. Select from [auto, raw].
            Default is 'auto'.
        opacity_colorspace (string):
            Colorspace of texture, if provided. Select from [auto, raw].
            Default is 'auto'.
        ior_colorspace (string):
            Colorspace of texture, if provided. Select from [auto, raw]. Default is 'auto'.
        specular_colorspace (string):
            Colorspace of texture, if provided. Select from [auto, raw, sRGB].
            Default is 'auto'.
        normals_colorspace (string):
            Colorspace of texture, if provided. Select from [auto, raw, sRGB].
            Default is 'auto'.
        displacement_colorspace (string):
            Colorspace of texture, if provided. Select from [auto, raw].
            Default is 'auto'.
        shaders (dict):
            Dictionary mapping a shader name to a reader and writer function.
            (Currently cannot be set).
    """
    def __init__(
        self,
        diffuse_color=(0.5, 0.5, 0.5),
        roughness_value=0.5,
        metallic_value=0.,
        clearcoat_value=0.,
        clearcoat_roughness_value=0.01,
        opacity_value=1.0,
        opacity_threshold=0.0,
        ior_value=1.5,
        specular_color=(0.0, 0.0, 0.0),
        displacement_value=0.,
        diffuse_texture=None,
        roughness_texture=None,
        metallic_texture=None,
        clearcoat_texture=None,
        clearcoat_roughness_texture=None,
        opacity_texture=None,
        ior_texture=None,
        specular_texture=None,
        normals_texture=None,
        displacement_texture=None,
        is_specular_workflow=False,
        diffuse_colorspace='auto',
        roughness_colorspace='auto',
        metallic_colorspace='auto',
        clearcoat_colorspace='auto',
        clearcoat_roughness_colorspace='auto',
        opacity_colorspace='auto',
        ior_colorspace='auto',
        specular_colorspace='auto',
        normals_colorspace='auto',
        displacement_colorspace='auto',
        name=''
    ):
        super().__init__(name)
        self.diffuse_color = diffuse_color
        self.roughness_value = roughness_value
        self.metallic_value = metallic_value
        self.clearcoat_value = clearcoat_value
        self.clearcoat_roughness_value = clearcoat_roughness_value
        self.opacity_value = opacity_value
        self.opacity_threshold = opacity_threshold
        self.ior_value = ior_value
        self.specular_color = specular_color
        self.displacement_value = displacement_value
        self.diffuse_texture = diffuse_texture
        self.roughness_texture = roughness_texture
        self.metallic_texture = metallic_texture
        self.clearcoat_texture = clearcoat_texture
        self.clearcoat_roughness_texture = clearcoat_roughness_texture
        self.opacity_texture = opacity_texture
        self.ior_texture = ior_texture
        self.specular_texture = specular_texture
        self.normals_texture = normals_texture
        self.displacement_texture = displacement_texture
        self.diffuse_colorspace = diffuse_colorspace
        self.roughness_colorspace = roughness_colorspace
        self.metallic_colorspace = metallic_colorspace
        self.clearcoat_colorspace = clearcoat_colorspace
        self.clearcoat_roughness_colorspace = clearcoat_roughness_colorspace
        self.opacity_colorspace = opacity_colorspace
        self.ior_colorspace = ior_colorspace
        self.specular_colorspace = specular_colorspace
        self.normals_colorspace = normals_colorspace
        self.displacement_colorspace = displacement_colorspace
        self.is_specular_workflow = is_specular_workflow

        self.shaders = {
            'UsdPreviewSurface': {
                'reader': self._read_usd_preview_surface,
                'writer': self._write_usd_preview_surface,
            }
        }

    def write_to_usd(self, file_path, scene_path, bound_prims=None, time=None,
                     texture_dir='', texture_file_prefix='', shader='UsdPreviewSurface'):
        r"""Write material to USD.
        Textures will be written to disk in with filename in the form:
        `{usd_file_path}/{texture_dir}/{texture_file_prefix}{attr}.png` where `attr` is one of
        [`diffuse`, `roughness`, `metallic`, `specular`, `normals`].

        Args:
            file_path (str): Path to usd file (\*.usd, \*.usda).
            scene_path (str): Absolute path of material within the USD file scene. Must be a valid ``Sdf.Path``.
            bound_prims (list of Usd.Prim, optional): If provided, bind material to each prim.
            time (convertible to float, optional): Positive integer defining the time at which the supplied parameters
                correspond to.
            texture_dir (str, optional): Subdirectory to store texture files. If not provided, texture files will be
                saved in the same directory as the USD file specified by `file_path`.
            texture_file_prefix (str, optional): String to be prepended to the filename of each texture file.
            shader (str, optional): Name of shader to write. If not provided, use UsdPreviewSurface.
        """
        assert os.path.splitext(file_path)[1] in ['.usd', '.usda'], f'Invalid file path "{file_path}".'
        assert shader in self.shaders, f'Shader {shader} is not support. Choose from {list(self.shaders.keys())}.'
        if os.path.exists(file_path):
            stage = Usd.Stage.Open(file_path)
        else:
            stage = create_stage(file_path)
        if time is None:
            time = Usd.TimeCode.Default()

        writer = self.shaders[shader]['writer']
        return writer(stage, scene_path, bound_prims, time, texture_dir, texture_file_prefix)

    def _write_usd_preview_surface(self, stage, scene_path, bound_prims,
                                   time, texture_dir, texture_file_prefix):
        """Write a USD Preview Surface material."""
        usd_file_path = stage.GetRootLayer().realPath
        texture_dir = Path(texture_dir).as_posix()
        material = UsdShade.Material.Define(stage, scene_path)

        shader = UsdShade.Shader.Define(stage, f'{scene_path}/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')

        # Create Inputs
        diffuse_input = shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f)
        roughness_input = shader.CreateInput('roughness', Sdf.ValueTypeNames.Float)
        specular_input = shader.CreateInput('specularColor', Sdf.ValueTypeNames.Color3f)
        metallic_input = shader.CreateInput('metallic', Sdf.ValueTypeNames.Float)
        clearcoat_input = shader.CreateInput('clearcoat', Sdf.ValueTypeNames.Float)
        clearcoat_roughness_input = shader.CreateInput('clearcoatRoughness', Sdf.ValueTypeNames.Float)
        opacity_input = shader.CreateInput('opacity', Sdf.ValueTypeNames.Float)
        opacity_threshold_input = shader.CreateInput('opacityThreshold', Sdf.ValueTypeNames.Float)
        ior_input = shader.CreateInput('ior', Sdf.ValueTypeNames.Float)
        normal_input = shader.CreateInput('normal', Sdf.ValueTypeNames.Normal3f)
        is_specular_workflow_input = shader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int)
        displacement_input = shader.CreateInput('displacement', Sdf.ValueTypeNames.Float)

        # Set constant values
        if self.diffuse_color is not None:
            diffuse_input.Set(tuple(self.diffuse_color), time=time)
        if self.roughness_value is not None:
            roughness_input.Set(self.roughness_value, time=time)
        if self.specular_color is not None:
            specular_input.Set(tuple(self.specular_color), time=time)
        if self.metallic_value is not None:
            metallic_input.Set(self.metallic_value, time=time)
        if self.clearcoat_value is not None:
            clearcoat_input.Set(self.clearcoat_value, time=time)
        if self.clearcoat_roughness_value is not None:
            clearcoat_roughness_input.Set(self.clearcoat_roughness_value, time=time)
        if self.opacity_value is not None:
            opacity_input.Set(self.opacity_value, time=time)
        if self.opacity_threshold is not None:
            opacity_threshold_input.Set(self.opacity_threshold, time=time)
        if self.ior_value is not None:
            ior_input.Set(self.ior_value, time=time)
        is_specular_workflow_input.Set(int(self.is_specular_workflow), time=time)
        if self.displacement_value is not None:
            displacement_input.Set(self.displacement_value, time=time)

        # Export textures and connect textures to shader
        usd_dir = os.path.dirname(usd_file_path)
        if self.diffuse_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f'{texture_file_prefix}diffuse.png')
            self._write_image(self.diffuse_texture, posixpath.join(usd_dir, rel_filepath))
            texture = self._add_texture_shader(
                stage, f'{scene_path}/diffuse_texture', rel_filepath, time=time, channels_out=3,
                colorspace=self.diffuse_colorspace)
            inputTexture = texture.CreateOutput('rgb', Sdf.ValueTypeNames.Color3f)
            diffuse_input.ConnectToSource(inputTexture)
        if self.roughness_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f'{texture_file_prefix}roughness.png')
            self._write_image(self.roughness_texture, posixpath.join(usd_dir, rel_filepath))
            texture = self._add_texture_shader(
                stage, f'{scene_path}/roughness_texture', rel_filepath, time=time, channels_out=1,
                colorspace=self.roughness_colorspace)
            inputTexture = texture.CreateOutput('r', Sdf.ValueTypeNames.Float)
            roughness_input.ConnectToSource(inputTexture)
        if self.specular_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f'{texture_file_prefix}specular.png')
            self._write_image(self.specular_texture, posixpath.join(usd_dir, rel_filepath))
            texture = self._add_texture_shader(
                stage, f'{scene_path}/specular_texture', rel_filepath, time=time, channels_out=3,
                colorspace=self.specular_colorspace)
            inputTexture = texture.CreateOutput('rgb', Sdf.ValueTypeNames.Color3f)
            specular_input.ConnectToSource(inputTexture)
        if self.metallic_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f'{texture_file_prefix}metallic.png')
            self._write_image(self.metallic_texture, posixpath.join(usd_dir, rel_filepath))
            texture = self._add_texture_shader(
                stage, f'{scene_path}/metallic_texture', rel_filepath, time=time, channels_out=1,
                colorspace=self.metallic_colorspace)
            inputTexture = texture.CreateOutput('r', Sdf.ValueTypeNames.Float)
            metallic_input.ConnectToSource(inputTexture)
        if self.clearcoat_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f'{texture_file_prefix}clearcoat.png')
            self._write_image(self.clearcoat_texture, posixpath.join(usd_dir, rel_filepath))
            texture = self._add_texture_shader(
                stage, f'{scene_path}/clearcoat_texture', rel_filepath, time=time, channels_out=1,
                colorspace=self.clearcoat_colorspace)
            inputTexture = texture.CreateOutput('r', Sdf.ValueTypeNames.Float)
            clearcoat_input.ConnectToSource(inputTexture)
        if self.clearcoat_roughness_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f'{texture_file_prefix}clearcoat_roughness.png')
            self._write_image(self.clearcoat_roughness_texture, posixpath.join(usd_dir, rel_filepath))
            texture = self._add_texture_shader(
                stage, f'{scene_path}/clearcoat_roughness_texture', rel_filepath, time=time, channels_out=1,
                colorspace=self.clearcoat_roughness_colorspace)
            inputTexture = texture.CreateOutput('r', Sdf.ValueTypeNames.Float)
            clearcoat_roughness_input.ConnectToSource(inputTexture)
        if self.opacity_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f'{texture_file_prefix}opacity.png')
            self._write_image(self.opacity_texture, posixpath.join(usd_dir, rel_filepath))
            texture = self._add_texture_shader(
                stage, f'{scene_path}/opacity_texture', rel_filepath, time=time, channels_out=1,
                colorspace=self.opacity_colorspace)
            inputTexture = texture.CreateOutput('r', Sdf.ValueTypeNames.Float)
            opacity_input.ConnectToSource(inputTexture)
        if self.ior_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f'{texture_file_prefix}ior.png')
            self._write_image(self.ior_texture, posixpath.join(usd_dir, rel_filepath))
            texture = self._add_texture_shader(
                stage, f'{scene_path}/ior_texture', rel_filepath, time=time, channels_out=1,
                colorspace=self.ior_colorspace)
            inputTexture = texture.CreateOutput('r', Sdf.ValueTypeNames.Float)
            ior_input.ConnectToSource(inputTexture)
        if self.normals_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f'{texture_file_prefix}normals.png')
            self._write_image(((self.normals_texture + 1.) / 2.), posixpath.join(usd_dir, rel_filepath))
            texture = self._add_texture_shader(
                stage, f'{scene_path}/normals_texture', rel_filepath, time=time, channels_out=3,
                colorspace=self.normals_colorspace)
            inputTexture = texture.CreateOutput('rgb', Sdf.ValueTypeNames.Normal3f)
            normal_input.ConnectToSource(inputTexture)
        if self.displacement_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f'{texture_file_prefix}displacement.png')
            self._write_image(self.displacement_texture, posixpath.join(usd_dir, rel_filepath))
            texture = self._add_texture_shader(
                stage, f'{scene_path}/displacement_texture', rel_filepath, time=time, channels_out=1,
                colorspace=self.displacement_colorspace)
            inputTexture = texture.CreateOutput('r', Sdf.ValueTypeNames.Float)
            displacement_input.ConnectToSource(inputTexture)

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

    def _add_texture_shader(self, stage, path, texture_path, time, channels_out=3, scale=None, bias=None,
                            colorspace=None):
        assert channels_out > 0 and channels_out <= 4
        texture = UsdShade.Shader.Define(stage, path)
        texture.CreateIdAttr('UsdUVTexture')
        texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texture_path, time=time)
        if scale is not None:
            texture.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(scale, time=time)
        if bias is not None:
            texture.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(bias, time=time)
        if colorspace is not None:
            texture.CreateInput('colorspace', Sdf.ValueTypeNames.Token).Set(colorspace, time=time)

        channels = ['r', 'b', 'g', 'a']
        for channel in channels[:channels_out]:
            texture.CreateOutput(channel, Sdf.ValueTypeNames.Float)

        return texture

    @staticmethod
    def _read_image(path, colorspace="auto"):
        """

        From https://graphics.pixar.com/usd/release/wp_usdpreviewsurface.html
        `colorspace`: Flag indicating the color space in which the source texture is encoded. Possible Values:
            raw : Use texture data as it was read from the texture and do not mark it as using a specific color space.
            sRGB : Mark texture as sRGB when reading.
            auto : Check for gamma/color space metadata in the texture file itself; if metadata is indicative of sRGB,
                mark texture as sRGB . If no relevant metadata is found, mark texture as sRGB if it is either 8-bit and
                has 3 channels or if it is 8-bit and has 4 channels. Otherwise, do not mark texture as sRGB and use
                texture data as it was read from the texture.


        """
        if colorspace.lower() not in ['auto', 'srgb', 'raw']:
            raise MaterialLoadError(f'Colorspace {colorspace} is not supported. Valid values are [auto, sRGB, raw]')
        if not os.path.exists(path):
            raise MaterialLoadError(f'No such image file: `{path}`')

        img = Image.open(str(path))
        return ((torch.FloatTensor(img.getdata())).reshape(*img.size, -1) / 255.).permute(2, 0, 1)

    @staticmethod
    def _write_image(img_tensor, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
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
            time (convertible to float, optional): Positive integer indicating the time at which to retrieve parameters.
        """
        if time is None:
            time = Usd.TimeCode.Default()
        if texture_path is None:
            texture_file_path = os.path.dirname(file_path)
        else:
            usd_dir = os.path.dirname(file_path)
            texture_file_path = posixpath.join(usd_dir, texture_path)
        stage = Usd.Stage.Open(file_path)
        material = UsdShade.Material(stage.GetPrimAtPath(scene_path))
        assert material

        surface_shader = material.GetSurfaceOutput().GetConnectedSource()[0]
        shader = UsdShade.Shader(surface_shader)
        if shader.GetImplementationSourceAttr().Get(time=time) == 'id':
            shader_name = UsdShade.Shader(surface_shader).GetShaderId()
        else:
            raise NotImplementedError

        material_params = _get_shader_parameters(surface_shader, time)

        reader = self.shaders[shader_name]['reader']
        return reader(material_params, texture_file_path, time)

    @classmethod
    def _read_usd_preview_surface(cls, shader_parameters, texture_file_path, time):
        """Read UsdPreviewSurface material."""
        texture_file_path = Path(texture_file_path).as_posix()
        params = {}

        def _read_data(data):
            if 'value' in data:
                if hasattr(data['value'], '__len__') and len(data['value']) > 1:
                    return tuple(data['value']), True
                else:
                    return data['value'], True
            elif 'file' in data and data['file']:
                fp = posixpath.join(texture_file_path, data['file'].get('value', Sdf.AssetPath()).path)
                if 'colorspace' in data:
                    colorspace = data['colorspace']['value']
                else:
                    colorspace = 'auto'
                try:
                    return cls._read_image(fp, colorspace=colorspace), False
                except MaterialLoadError:
                    warnings.warn(f'An error was encountered while processing the data {data["file"]}.')
                    if 'fallback' in data:
                        return data['fallback']['value'], True
            return None, False

        for name, data in shader_parameters.items():
            if 'diffuse' in name.lower() or 'albedo' in name.lower():
                output, is_value = _read_data(data)
                params[f'diffuse_{["texture", "color"][is_value]}'] = output
                if 'colorspace' in data:
                    params['diffuse_colorspace'] = data['colorspace']['value']
            elif 'roughness' in name.lower() and 'clearcoat' not in name.lower():
                output, is_value = _read_data(data)
                params[f'roughness_{["texture", "value"][is_value]}'] = output
                if 'colorspace' in data:
                    params['roughness_colorspace'] = data['colorspace']['value']
            elif 'metallic' in name.lower():
                output, is_value = _read_data(data)
                params[f'metallic_{["texture", "value"][is_value]}'] = output
                if 'colorspace' in data:
                    params['metallic_colorspace'] = data['colorspace']['value']
            elif 'clearcoat' in name.lower():
                output, is_value = _read_data(data)
                params[f'clearcoat_{["texture", "value"][is_value]}'] = output
                if 'colorspace' in data:
                    params['clearcoat_colorspace'] = data['colorspace']['value']
            elif 'clearcoat_roughness' in name.lower():
                output, is_value = _read_data(data)
                params[f'clearcoat_roughness_{["texture", "value"][is_value]}'] = output
                if 'colorspace' in data:
                    params['clearcoat_roughness_colorspace'] = data['colorspace']['value']
            elif 'opacitythreshold' in name.lower():
                output, _ = _read_data(data)
                params['opacity_threshold'] = output
            elif 'opacity' in name.lower():
                output, is_value = _read_data(data)
                params[f'opacity_{["texture", "value"][is_value]}'] = output
                if 'colorspace' in data:
                    params['opacity_colorspace'] = data['colorspace']['value']
            elif 'ior' in name.lower():
                output, is_value = _read_data(data)
                params[f'ior_{["texture", "value"][is_value]}'] = output
                if 'colorspace' in data:
                    params['ior_colorspace'] = data['colorspace']['value']
            elif 'specular' in name.lower():
                output, is_value = _read_data(data)
                if 'workflow' in name.lower():
                    params['is_specular_workflow'] = bool(output)
                else:
                    params[f'specular_{["texture", "color"][is_value]}'] = output
                    if 'colorspace' in data:
                        params['specular_colorspace'] = data['colorspace']['value']
            elif 'normal' in name.lower():
                output, is_value = _read_data(data)
                if not is_value:
                    params['normals_texture'] = output * 2. - 1.
                    if 'colorspace' in data:
                        params['normals_colorspace'] = data['colorspace']['value']
            if 'displacement':
                output, is_value = _read_data(data)
                params[f'displacement_{["texture", "value"][is_value]}'] = output
                if 'colorspace' in data:
                    params['displacement_colorspace'] = data['colorspace']['value']
        return cls(**params)


def process_materials_and_assignments(materials_dict, material_assignments_dict, error_handler, num_faces,
                                      error_context_str=''):
    """Converts dictionary style materials and assignments to final format (see args/return values).

    Args:
        materials_dict (dict of str to dict): mapping from material name to material parameters
        material_assignments_dict (dict of str to torch.LongTensor): mapping from material name to either
           1) a K x 2 tensor with start and end face indices of the face ranges assigned to that material or
           2) a K, tensor with face indices assigned to that material
        error_handler: handler able to handle MaterialNotFound error - error can be thrown, ignored, or the
            handler can return a dummy material for material not found (if this is not the case, assignments to
            non-existent materials will be lost), e.g. obj.create_missing_materials_error_handler.
        num_faces: total number of faces in the model
        error_context_str (str): any extra info to attach to thrown errors

    Returns:
        (tuple) of:

        - **materials** (list): list of material parameters, sorted alphabetically by their name
        - **material_assignments** (torch.ShortTensor): of shape `(\text{num_faces},)` containing index of the
            material (in the above list) assigned to the corresponding face, or `-1` if no material was assigned.
    """
    def _try_to_set_name(generated_material, material_name):
        if isinstance(generated_material, Mapping):
            generated_material['material_name'] = material_name
        else:
            try:
                generated_material.material_name = material_name
            except Exception as e:
                warnings.warn(f'Cannot set dummy material_name: {e}')

    # Check that all assigned materials exist and if they don't we create a dummy material
    missing_materials = []
    for mat_name in material_assignments_dict.keys():
        if mat_name not in materials_dict:
            dummy_material = error_handler(
                MaterialNotFoundError(f"'Material {mat_name}' not found, but referenced. {error_context_str}"))

            # Either create dummy material or remove assignment
            if dummy_material is not None:
                _try_to_set_name(dummy_material, mat_name)
                materials_dict[mat_name] = dummy_material
            else:
                missing_materials.append(mat_name)

    # Ignore assignments to missing materials (unless handler created dummy material)
    for mat_name in missing_materials:
        del material_assignments_dict[mat_name]

    material_names = sorted(materials_dict.keys())
    materials = [materials_dict[name] for name in material_names]  # Alphabetically ordered materials
    material_assignments = torch.zeros((num_faces,), dtype=torch.int16) - 1

    # Process material assignments to use material indices instead
    for name, values in material_assignments_dict.items():
        mat_idx = material_names.index(name)  # Alphabetically sorted material

        if len(values.shape) == 1:
            indices = values
        else:
            assert len(values.shape) == 2 and values.shape[-1] == 2, \
                f'Unxpected shape {values.shape} for material assignments for material {name} ' \
                f'(expected (K,) or (K, 2)). {error_context_str}'
            # Rewrite (K, 2) tensor of (face_idx_start, face_idx_end] to (M,) tensor of face_idx
            indices = torch.cat(
                [torch.arange(values[r, 0], values[r, 1], dtype=torch.long) for r in range(values.shape[0])])

        # Use face indices as index to set material_id in face-aligned material assignments
        material_assignments[indices] = mat_idx

    return materials, material_assignments


MaterialManager.register_usd_reader('UsdPreviewSurface', PBRMaterial._read_usd_preview_surface)
