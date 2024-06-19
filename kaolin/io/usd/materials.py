# Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections.abc import Callable, Mapping
import inspect
import logging
import os
import pathlib
import re
import posixpath
import warnings

try:
    from pxr import UsdShade, Sdf, Usd
except ImportError:
    warnings.warn('Warning: module pxr not found', ImportWarning)

import kaolin.io.utils
from kaolin.io.materials import MaterialLoadError, MaterialWriteError, MaterialNotSupportedError, MaterialError
from kaolin.render.materials import Material, PBRMaterial
from .utils import _get_stage_from_maybe_file, get_scene_paths


logging.getLogger("PIL.PngImagePlugin").propagate = False


# TODO: ensure this can also work for absolute texture paths
def import_material(file_path_or_stage, scene_path, texture_path=None, time=None) -> Material:
    r"""Read material from file and return a PyTorch Material object (e.g. see :class:`PBRMaterial <PBRMaterial>`).
        The shader used must have a corresponding registered reader function. Currently supports `UsdPreviewSurface`.

        Args:
            file_path_or_stage (str, Usd.Stage):
                Path to usd file (`\*.usd`, `\*.usda`, `\*.usdc`) or :class:`Usd.Stage`.
            scene_path (str): Absolute path of UsdShade.Material prim within the USD file scene.
                Must be a valid ``Sdf.Path``.
            texture_path (str, optional): Path to textures directory. By default, the textures will be assumed to be
                under the same directory as the file specified by `file_path`.
            time (convertible to float, optional): Optional for reading USD files. Positive integer indicating the
                time at which to retrieve parameters.

        Returns:
            (Material): Material object determined by the corresponding reader function.

        Raises:
            MaterialLoadError: if any error is encountered during load.
    """
    stage = _get_stage_from_maybe_file(file_path_or_stage)

    if texture_path is None or not os.path.isabs(texture_path):
        usd_dir = os.path.dirname(stage.GetRootLayer().realPath)
        if texture_path is None:
            texture_path = usd_dir
        else:
            texture_path = posixpath.join(usd_dir, texture_path)

    if scene_path is None or not Sdf.Path(scene_path):
        raise MaterialLoadError(f'The scene_path "{scene_path}" provided is invalid.')

    try:
        material = UsdShade.Material(stage.GetPrimAtPath(scene_path))
        return UsdMaterialIoManager.read_material(material, texture_path=texture_path, time=time)
    except Exception as e:
        raise MaterialLoadError(f'Failed to read UsdShade.Material from {scene_path} with exception {e}')


def export_material(pbr_material: Material, file_path_or_stage, scene_path=None, texture_path=None,
                    bound_prims=None, texture_file_prefix=None, shader_name=None, time=None,
                    overwrite_textures=False):
    r"""Exports a PyTorch Material object (e.g. see :class:`PBRMaterial <PBRMaterial>`) to USD.
        The shader used must have a corresponding registered writer function. Currently supports `UsdPreviewSurface`.

        Args:
            pbr_material: instance of `Material` class with a writer registered
            file_path_or_stage (str, Usd.Stage):
                Path to usd file (`\*.usd`, `\*.usda`, `\*.usdc`) or :class:`Usd.Stage`.
            scene_path (str, optional): Absolute path to use for UsdShade.Material prim within the USD file scene.
                If not set, will suggest a scene_path that is not yet present in the scene
                (for example "/World/Looks/material_name_0", avoiding collisions); the scene path used will be returned.
            texture_path (str, optional): Directory where to save textures. Will use the relative version of this
                path to save in USD. If not set, will use default "textures" directory relative to the USD path.
            bound_prims (list of Usd.Prim, optional): If provided, binds material to each prim.
            texture_file_prefix (str, optional): Prefix to use for saved texture files.
            shader_name (str, optional): Will be used to find appropriate writer. By default set to
                `pbr_material.shader_name`, e.g. `UsdPreviewSurface`.
            time (convertible to float, optional): Optional for writing USD files. Positive integer indicating the
                time at which to set parameters.
            overwrite_textures (bool): set to True to overwrite existing image files when writing textures; if False
                (default) will add index to filename to avoid conflicts.

        Returns:
            (str, UsdShade.Material): scene_path where material was saved (e.g. if auto-set), and USD Material object.

        Raises:
            MaterialWriteError: if any error is encountered during write.
    """
    stage = _get_stage_from_maybe_file(file_path_or_stage, check_exists=False)
    usd_dir = os.path.dirname(stage.GetRootLayer().realPath)

    if scene_path is None:
        scene_path = _suggest_new_material_scene_path(pbr_material.material_name, stage)

    if shader_name is None:
        shader_name = pbr_material.shader_name

    if texture_path is None:
        relative_texture_dir = 'textures'
    elif os.path.isabs(texture_path):
        relative_texture_dir = os.path.relpath(texture_path, usd_dir)
    else:
        relative_texture_dir = texture_path

    if texture_file_prefix is None:
        texture_file_prefix = f'{scene_path.split("/")[-1]}_'  # Last value of the path

    try:
        usd_material = UsdMaterialIoManager.write_material(
            pbr_material, stage=stage, scene_path=scene_path, shader_name=shader_name,
            time=time, relative_texture_dir=relative_texture_dir,
            texture_file_prefix=texture_file_prefix, overwrite_textures=overwrite_textures)

        if bound_prims is not None:
            for prim in bound_prims:
                binding_api = UsdShade.MaterialBindingAPI(prim)
                binding_api.Bind(usd_material)
        stage.Save()
    except Exception as e:
        raise MaterialWriteError(f'Failed to write material with exception {e}')

    return scene_path, usd_material


# TODO: can we get all material scene_paths and implement import_materials
#  import_materials (file_path_or_stage, scene_paths=None, texture_path=None, times=None)

def _suggest_new_material_scene_path(material_name, stage, base_path='/World/Looks'):
    existing_paths = [str(x) for x in get_scene_paths(stage)] + [posixpath.join(base_path, '/material')]  # enforce numbered materials
    if material_name is not None and len(material_name) > 0:
        candidate = re.sub("[^a-zA-Z0-9/]+", "_", material_name)
    else:
        candidate = 'material'

    if "/" not in candidate:  # Assume just a name, not nested under Looks or similar
        candidate = posixpath.join(base_path, candidate)

    idx = 0
    candidate_prefix = candidate
    while candidate in existing_paths:
        candidate = f'{candidate_prefix}_{idx}'
        idx += 1

    return candidate


def _check_callable_with_args(fn, num_args, required_args=None):
    if not isinstance(fn, Callable):
        raise ValueError('The supplied reader/writer must be a callable function.')

    # Validate function signature
    actual_params = list(inspect.signature(fn).parameters)

    if num_args is not None and len(actual_params) != num_args:
        raise ValueError(f'The supplied reader/writer has {len(actual_params)} ({num_args} expected)')

    if required_args is not None:
        for ar in required_args:
            if ar not in actual_params:
                raise ValueError(f'The supplied reader/writer is missing argument with name {ar}; '
                                 f'found arguments {actual_params}')


class UsdMaterialIoManager:
    """Material management utility for USD Material I/O, which allows expanding the USD material support.
    Allows material reader functions to be mapped against specific shaders. This allows USD import functions
    to determine if a reader is available, which material reader to use and which material representation to wrap the
    output with.

    Also supports registering writers, with a more complicated signature due to the intricacies of dealing
    with saving / overwriting texture files. See :meth:`~UsdMaterialIoManager.register_usd_writer`.

    Default registered readers / writers:

    - UsdPreviewSurface: Import material with shader id `UsdPreviewSurface`. All parameters are supported,
      including textures. See https://graphics.pixar.com/usd/release/wp_usdpreviewsurface.html for more details
      on available material parameters.

    Example:
        >>> # Register a new USD reader for mdl `MyCustomPBR`
        >>> from kaolin.io.usd import materials
        >>> dummy_reader = lambda params_dict, texture_path: UsdShade.Material()
        >>> materials.MaterialManager.register_usd_reader('MyCustomPBR', dummy_reader)
    """
    _usd_readers = {}
    _usd_writers = {}

    @classmethod
    def register_usd_reader(cls, shader_name, reader_fn):
        """Register a shader reader function that will be used during USD material import.

        Args:
            shader_name (str): Name of the shader
            reader_fn (Callable): Function that will be called to read shader attributes. The function must take as
                input a dictionary of input parameters and a string representing the texture path
                `(UsdShade.Shader, texture_path, time)` and typically return a `Material`
        """
        if shader_name in cls._usd_readers:
            warnings.warn(f'Shader {shader_name} is already registered. Overwriting previous definition.')

        _check_callable_with_args(reader_fn, 3)
        cls._usd_readers[shader_name] = reader_fn

    @classmethod
    def register_usd_writer(cls, shader_name, writer_fn):
        """Register a shader reader function that will be used during USD material import.

        Args:
            shader_name (str): Name of the shader
            writer_fn (Callable): Function that will be called to write a material. The function must take as
                input named parameters `(material_object, stage, scene_path, time, write_texture_by_basename_fn)`,
                where `write_texture_by_basename_fn` is a callable that takes `(pytorch_image, texture_file_basename)`,
                writes texture to file, and returns its relative path. Such a callable can be easily created using
                `TextureExporter`. The `writer_fn` must return created `UsdShade.Material`.
        """
        if shader_name in cls._usd_writers:
            warnings.warn(f'Shader {shader_name} is already registered. Overwriting previous definition.')

        _check_callable_with_args(
            writer_fn, 5, ['stage', 'scene_path', 'time', 'write_texture_by_basename_fn'])
        cls._usd_writers[shader_name] = writer_fn

    @classmethod
    def read_material(cls, usd_material, texture_path, time=None):
        r"""Read USD material into internal PyTorch material representation. The type returned will be the
        type returned by the registered reader function, selected based on shader name specified in the
        usd_material.

        Args:
            usd_material (UsdShade.Material): Valid USD Material prim
            texture_path (str, optional): Path to textures directory. If the USD has absolute paths
                to textures, set to an empty string. By default, the textures will be assumed to be
                under the same directory as the USD specified by `file_path`.
            time (convertible to float, optional): Positive integer indicating the time at which to retrieve parameters.

        Returns:
            Material object determined by the corresponding reader function, or raw shader properties as
            a dictionary.
        """
        if time is None:
            time = Usd.TimeCode.Default()

        if not UsdShade.Material(usd_material):
            raise MaterialLoadError(f'The material `{usd_material}` is not a valid UsdShade.Material object.')

        for surface_output in usd_material.GetSurfaceOutputs():
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
                warnings.warn(f'A reader for the material defined by `{usd_material}` is not yet implemented.')

            if shader_name not in cls._usd_readers:
                warnings.warn('No registered readers were able to process the material '
                              f'`{usd_material}` with shader `{shader_name}`.')
                params = _get_shader_parameters(surface_shader, time)
                params['material_name'] = str(usd_material.GetPath())
                return params

            reader = cls._usd_readers[shader_name]
            res = reader(surface_shader, texture_path, time)
            try:
                res.material_name = str(usd_material.GetPath())
            except Exception as e:
                pass
            return res

        raise MaterialError(f'Error processing material {usd_material}')

    @classmethod
    def write_material(cls, pbr_material: Material, stage, scene_path, shader_name, relative_texture_dir,
                       texture_file_prefix='', time=None, overwrite_textures=False, image_extension='.png'):
        r"""Writes internal PyTorch based material to USD.
        Textures will be written to disk in with filename in the form:
        `usd_file_path/{relative_texture_dir}/{texture_file_prefix}{attr}{image_extension}` where `attr` is
        attributes like `diffuse`, `roughness`, `metallic`, `specular`, `normals`, etc.

        Args:
            pbr_material (Material): material object that has a registered writer.
            stage (Usd.Stage): Path to usd file (\*.usd, \*.usda).
            scene_path (str): Absolute path of material within the USD file scene. Must be a valid ``Sdf.Path``.
            shader_name (str): Shader name with a registered writer for the input material type.
            relative_texture_dir (str): relative texture directory; must be relative to USD file root.
            texture_file_prefix (str): optional prefix to add to all texture files written (not a path, part of basename)
            time (convertible to float, optional): Positive integer defining the time at which the supplied parameters
                correspond to.
            overwrite_textures (bool): set to True to overwrite existing image files when writing textures; if False
                (default) will add index to filename to avoid any conflicts.
            image_extension (str): extension, including dot, to use for image files, such as ".jpg", ".png" (default).
        """
        if time is None:
            time = Usd.TimeCode.Default()

        writer = cls._usd_writers.get(shader_name)
        if writer is None:
            raise MaterialNotSupportedError(f'Writer for shader type {shader_name} has not been registered. '
                                            f'Available shaders: {cls._usd_writers}')

        texture_exporter = kaolin.io.utils.TextureExporter(
            os.path.dirname(stage.GetRootLayer().realPath), relative_texture_dir,
            file_prefix=texture_file_prefix, image_extension=image_extension,
            overwrite_files=overwrite_textures)

        # if stage.GetPrimAtPath(scene_path):
        #     logging.warning(f'Overwriting existing material at path {scene_path}')
        #     stage.RemovePrim(scene_path)

        material = writer(pbr_material, stage=stage, scene_path=scene_path, time=time,
                          write_texture_by_basename_fn=texture_exporter)

        if material is None:
            raise MaterialWriteError(f'Writer for shader {shader_name} returned None')

        return material


def _add_texture_shader(stage, path, texture_path, time, channels_out=3, scale=None, bias=None, colorspace=None):
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

    return kaolin.io.utils.read_image(path)


def _get_shader_parameters(shader, time):
    # Get shader parameters
    params = {}
    inputs = shader.GetInputs()
    for i in inputs:
        name = i.GetBaseName()
        params.setdefault(i.GetBaseName(), {})

        # Input can have both base value and connected input
        params[name].update({
            'value': i.Get(time=time),
            'type': i.GetTypeName().type,
            'docs': i.GetDocumentation(),
        })
        if UsdShade.ConnectableAPI.HasConnectedSource(i):
            connected_source = UsdShade.ConnectableAPI.GetConnectedSource(i)
            connected_inputs = connected_source[0].GetInputs()
            # TODO: Not sure this recursion actually works, needs to account
            # for the connected_source[1], which is the output channel such as 'r'
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
    return params


def read_usd_preview_surface(shader, texture_file_path, time):
    """Read UsdPreviewSurface material."""
    texture_file_path = pathlib.Path(texture_file_path).as_posix()

    shader_parameters = _get_shader_parameters(shader, time)
    params = {}

    def _read_data(data):
        value = None
        texture_value = None
        if 'value' in data:
            if hasattr(data['value'], '__len__') and len(data['value']) > 1:
                value = tuple(data['value'])
            else:
                value = data['value']
        if 'file' in data and data['file']:
            fp = data['file'].get('value', Sdf.AssetPath()).resolvedPath
            if not os.path.exists(fp):  # if relative path not already resolved, let's use relative texture dir
                fp = posixpath.join(texture_file_path, data['file'].get('value', Sdf.AssetPath()).path)
            if not os.path.exists(fp):  # if texture dir was passed in without accounting for relative paths set in USD
                fp = posixpath.join(texture_file_path, os.path.basename(data['file'].get('value', Sdf.AssetPath()).path))
            if 'colorspace' in data:
                colorspace = data['colorspace']['value']
            else:
                colorspace = 'auto'
            try:
                texture_value = _read_image(fp, colorspace=colorspace)
            except MaterialLoadError as e:
                # TODO: perhaps raise the error instead?
                warnings.warn(f'An error was encountered while processing the data {data["file"]}. "{e}"')
                if 'fallback' in data:
                    texture_value = data['fallback']['value']
                else:
                    raise e
        return value, texture_value

    def _process_input_arg(data, name, value_name='value'):
        value, texture_value = _read_data(data)
        if value is not None:
            params[f'{name}_{value_name}'] = value
        if texture_value is not None:
            params[f'{name}_texture'] = texture_value
        if 'colorspace' in data:
            params[f'{name}_colorspace'] = data['colorspace']['value']

    for full_name, data in shader_parameters.items():
        name = full_name.lower()
        if 'diffuse' in name or 'albedo' in name:
            _process_input_arg(data, 'diffuse', 'color')
        elif 'roughness' in name and 'clearcoat' not in name:
            _process_input_arg(data, 'roughness')
            # HACK: address TODO in _get_shader_parameters instead
            if 'roughness_texture' in params and params['roughness_texture'].shape[-1] > 1:
                params['roughness_texture'] = params['roughness_texture'][..., 1:2]
            # END OF HACK
        elif 'metallic' in name:
            _process_input_arg(data, 'metallic')
            # HACK: address TODO in _get_shader_parameters instead
            if 'metallic_texture' in params and params['metallic_texture'].shape[-1] > 2:
                params['metallic_texture'] = params['metallic_texture'][..., 2:3]
            # END OF HACK
        elif 'clearcoatroughness' in name:
            _process_input_arg(data, 'clearcoat_roughness')
        elif 'clearcoat' in name:
            _process_input_arg(data, 'clearcoat')
        elif 'opacitythreshold' in name:
            value, _ = _read_data(data)
            params['opacity_threshold'] = value
        elif 'opacity' in name:
            _process_input_arg(data, 'opacity')
        elif 'ior' in name:
            _process_input_arg(data, 'ior')
        elif 'specular' in name and 'workflow' in name:
            value, _ = _read_data(data)
            params['is_specular_workflow'] = bool(value)
        elif 'specular' in name and 'workflow' not in name:
            _process_input_arg(data, 'specular', 'color')
        elif 'normal' in name:
            _, texture_value = _read_data(data)
            if texture_value is not None:
                params['normals_texture'] = texture_value * 2. - 1.
                if 'colorspace' in data:
                    params['normals_colorspace'] = data['colorspace']['value']
        elif 'displacement' in name:
            _process_input_arg(data, 'displacement')
        else:
            logging.warning(f'Unprocessed shader parameter {name}')

    return PBRMaterial(**params)


def write_usd_preview_surface(pbr_material: PBRMaterial, stage, scene_path, write_texture_by_basename_fn,
                              time):
    r"""Writes USD Preview Surface Material to stage at the specified scene_path.

    Args:
        pbr_material (PBRMaterial): material definition
        stage (Usd.Stage): the stage where to write
        scene_path (str): scene path where to write the top level material
        write_texture_by_basename_fn (Callable): function that takes (PyTorch image, basename) and writes it
            to the correct location with correct extension on disk, handling overwriting/non-overwriting logic,
            and returning path that is relative and should be added as the reference in USD.
            See :class:`kaolin.io.utils.TextureExporter`.
        time (): optional time code

    Returns: UsdShade.Material
    """
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

    # TODO(cfujitsang): This doesn't seems to be taking into account colorspace
    # Set constant values
    if pbr_material.diffuse_color is not None:
        diffuse_input.Set(tuple(pbr_material.diffuse_color), time=time)
    if pbr_material.roughness_value is not None:
        roughness_input.Set(pbr_material.roughness_value.item(), time=time)
    if pbr_material.specular_color is not None:
        specular_input.Set(tuple(pbr_material.specular_color), time=time)
    if pbr_material.metallic_value is not None:
        metallic_input.Set(pbr_material.metallic_value.item(), time=time)
    if pbr_material.clearcoat_value is not None:
        clearcoat_input.Set(pbr_material.clearcoat_value.item(), time=time)
    if pbr_material.clearcoat_roughness_value is not None:
        clearcoat_roughness_input.Set(pbr_material.clearcoat_roughness_value.item(), time=time)
    if pbr_material.opacity_value is not None:
        opacity_input.Set(pbr_material.opacity_value.item(), time=time)
    if pbr_material.opacity_threshold is not None:
        opacity_threshold_input.Set(pbr_material.opacity_threshold.item(), time=time)
    if pbr_material.ior_value is not None:
        ior_input.Set(pbr_material.ior_value.item(), time=time)
    is_specular_workflow_input.Set(int(pbr_material.is_specular_workflow), time=time)
    if pbr_material.displacement_value is not None:
        displacement_input.Set(pbr_material.displacement_value.item(), time=time)

    def _process_texture(image, color_space, shader_input, name):
        if image is None:
            return
        if name == 'normals':
            image = (image + 1.) / 2.
        rel_filepath = write_texture_by_basename_fn(image, name)
        if shader_input.GetTypeName() in [Sdf.ValueTypeNames.Color3f, Sdf.ValueTypeNames.Normal3f]:
            num_channels = 3
            output_type = 'rgb'
        else:
            num_channels = 1
            output_type = 'r'
        texture = _add_texture_shader(
            stage, f'{scene_path}/{name}_texture', rel_filepath, time=time, channels_out=num_channels,
            colorspace=color_space)
        inputTexture = texture.CreateOutput(output_type, shader_input.GetTypeName())
        shader_input.ConnectToSource(inputTexture)

    for attr, shader_input in zip(
            ['diffuse', 'roughness', 'specular', 'metallic', 'normals',
             'clearcoat', 'clearcoat_roughness', 'opacity', 'ior', 'displacement'],
            [diffuse_input, roughness_input, specular_input, metallic_input, normal_input,
             clearcoat_input, clearcoat_roughness_input, opacity_input, ior_input, displacement_input]):
        _process_texture(getattr(pbr_material, f'{attr}_texture'), getattr(pbr_material, f'{attr}_colorspace'),
                         shader_input, attr)

    # create Usd Preview Surface Shader outputs
    shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    shader.CreateOutput('displacement', Sdf.ValueTypeNames.Token)

    # create material
    material.CreateSurfaceOutput().ConnectToSource(shader.GetOutput('surface'))
    material.CreateDisplacementOutput().ConnectToSource(shader.GetOutput('displacement'))

    return material


UsdMaterialIoManager.register_usd_reader('UsdPreviewSurface', read_usd_preview_surface)
UsdMaterialIoManager.register_usd_writer('UsdPreviewSurface', write_usd_preview_surface)
