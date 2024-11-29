# Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES.
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
import copy
import random
import torch

import kaolin.utils.testing


def _to_1d_tensor(data):
    if isinstance(data, torch.Tensor):
        return data.reshape(-1).float()
    elif data is None:
        return None
    else:
        return torch.tensor(data).reshape(-1).float()


class Material:
    """Abstract material definition class.
    Defines material inputs and methods to export material properties.
    """
    def __init__(self, name, shader_name):
        self.material_name = str(name)
        self.shader_name = str(shader_name)


class PBRMaterial(Material):
    """Defines a PBR material; allows storing rendering material properties imported from USD, gltf, obj,
    attempting to expose a consistent interface.
    Supports USD Preview Surface (https://graphics.pixar.com/usd/docs/UsdPreviewSurface-Proposal.html),
    a physically based surface material definition.

    Parameters:
        diffuse_color (tuple of floats):
            RGB values for `Diffuse` parameter (typically referred to as `Albedo`
            in a metallic workflow) in the range of `(0.0, 0.0, 0.0)` to `(1.0, 1.0, 1.0)`.
            Default value is None.
        roughness_value (float):
            Roughness value of specular lobe in range `0.0` to `1.0`. Default value is None.
        metallic_value (float):
            Metallic value, typically set to `0.0` for non-metallic and `1.0` for metallic materials. 
            Ignored if `is_specular_workflow` is `True`. Default value is None.
        clearcoat_value (float):
            Second specular lobe amount. Color is hardcoded to white. Default value is None.
        clearcoat_roughness_value (float):
            Roughness for the clearcoat specular lobe in the range `0.0` to `1.0`.
            The default value is None.
        opacity_value (float):
            Opacity, with `1.0` fully opaque and `0.0` as transparent with values within this range
            defining a translucent surface. Default value is None.
        opacity_threshold (float):
            Used to create cutouts based on the `opacity_value`. Surfaces with an opacity
            smaller than the `opacity_threshold` will be fully transparent. Default value is None.
            Note: The definition of `opacity` make conflict with the `transmission` field
            due to different shader conventions.
            Use either the one or the other according to your shader conventions.
        ior_value (float):
            Index of Refraction used with translucent objects and objects with specular components.
            Default value is None.
        specular_color (tuple of floats):
            RGB values for `Specular` lobe. Ignored if `is_specular_workflow` is `False`.
            Default value is None.
        displacement_value (float):
            Displacement in the direction of the normal. Default is None
        transmittance_value (float):
            The percentage of light that is transmitted through the surface of the material. Default is None
            Note: The definition of `transmission` make conflict with the `opacity` field
            due to different shader conventions.
            Use either the one or the other according to your shader conventions.
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
        transmittance_texture (torch.FloatTensor):
            Texture for the percentage of light that is transmitted through the surface of the material,
            of shape `(1, height, width)`.
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
        transmittance_colorspace (string):
            Colorspace of texture, if provided. Select from [auto, raw].
            Default is 'auto'.
        shaders (dict):
            Dictionary mapping a shader name to a reader and writer function.
            (Currently cannot be set).
    """

    __value_attributes = [
        'diffuse_color',
        'roughness_value',
        'metallic_value',
        'clearcoat_value',
        'clearcoat_roughness_value',
        'opacity_value',
        'opacity_threshold',
        'ior_value',
        'specular_color',
        'displacement_value',
        'transmittance_value'
    ]

    __texture_attributes = [
        'diffuse_texture',
        'roughness_texture',
        'metallic_texture',
        'clearcoat_texture',
        'clearcoat_roughness_texture',
        'opacity_texture',
        'ior_texture',
        'specular_texture',
        'normals_texture',
        'displacement_texture',
        'transmittance_texture'
    ]

    __colorspace_attributes = [
        'diffuse_colorspace',
        'roughness_colorspace',
        'metallic_colorspace',
        'clearcoat_colorspace',
        'clearcoat_roughness_colorspace',
        'opacity_colorspace',
        'ior_colorspace',
        'specular_colorspace',
        'normals_colorspace',
        'displacement_colorspace',
        'transmittance_colorspace'
    ]

    __misc_attributes = [
        'is_specular_workflow',
        'material_name',
        'shader_name'
    ]

    @classmethod
    def supported_tensor_attributes(cls):
        return cls.__texture_attributes + cls.__value_attributes

    __slots__ = __value_attributes + __texture_attributes + __colorspace_attributes + __misc_attributes

    def __init__(
        self,
        diffuse_color=None,
        roughness_value=None,
        metallic_value=None,
        clearcoat_value=None,
        clearcoat_roughness_value=None,
        opacity_value=None,
        opacity_threshold=None,
        ior_value=None,
        specular_color=None,
        displacement_value=None,
        transmittance_value=None,
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
        transmittance_texture=None,
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
        transmittance_colorspace='auto',
        material_name='',
        shader_name='UsdPreviewSurface'
    ):
        super().__init__(material_name, shader_name)
        self.diffuse_color = _to_1d_tensor(diffuse_color)
        assert self.diffuse_color is None or self.diffuse_color.shape == (3,)
        self.roughness_value = _to_1d_tensor(roughness_value)
        assert self.roughness_value is None or self.roughness_value.shape == (1,)
        self.metallic_value = _to_1d_tensor(metallic_value)
        assert self.metallic_value is None or self.metallic_value.shape == (1,)
        self.clearcoat_value = _to_1d_tensor(clearcoat_value)
        assert self.clearcoat_value is None or self.clearcoat_value.shape == (1,)
        self.clearcoat_roughness_value = _to_1d_tensor(clearcoat_roughness_value)
        assert self.clearcoat_roughness_value is None or self.clearcoat_roughness_value.shape == (1,)
        self.opacity_value = _to_1d_tensor(opacity_value)
        assert self.opacity_value is None or self.opacity_value.shape == (1,)
        self.opacity_threshold = _to_1d_tensor(opacity_threshold)
        assert self.opacity_threshold is None or self.opacity_threshold.shape == (1,)
        self.ior_value = _to_1d_tensor(ior_value)
        assert self.ior_value is None or self.ior_value.shape == (1,)
        self.specular_color = _to_1d_tensor(specular_color)
        if self.specular_color is not None:
            if self.specular_color.shape == (1,):
                self.specular_color = self.specular_color.repeat(3)
            else:
                assert self.specular_color.shape == (3,)
        self.displacement_value = _to_1d_tensor(displacement_value)
        assert self.displacement_value is None or self.displacement_value.shape == (1,)
        self.transmittance_value = _to_1d_tensor(transmittance_value)
        assert self.transmittance_value is None or self.transmittance_value.shape == (1,)
        assert diffuse_texture is None or diffuse_texture.dim() == 3
        self.diffuse_texture = diffuse_texture
        assert roughness_texture is None or roughness_texture.dim() == 3
        self.roughness_texture = roughness_texture
        assert metallic_texture is None or metallic_texture.dim() == 3
        self.metallic_texture = metallic_texture
        assert clearcoat_texture is None or clearcoat_texture.dim() == 3
        self.clearcoat_texture = clearcoat_texture
        assert clearcoat_roughness_texture is None or clearcoat_roughness_texture.dim() == 3
        self.clearcoat_roughness_texture = clearcoat_roughness_texture
        assert opacity_texture is None or opacity_texture.dim() == 3
        self.opacity_texture = opacity_texture
        assert ior_texture is None or ior_texture.dim() == 3
        self.ior_texture = ior_texture
        assert specular_texture is None or specular_texture.dim() == 3
        self.specular_texture = specular_texture
        assert normals_texture is None or normals_texture.dim() == 3
        self.normals_texture = normals_texture
        assert displacement_texture is None or displacement_texture.dim() == 3
        self.displacement_texture = displacement_texture
        assert transmittance_texture is None or transmittance_texture.dim() == 3
        self.transmittance_texture = transmittance_texture
        assert diffuse_colorspace in ['auto', 'raw', 'sRGB']
        self.diffuse_colorspace = diffuse_colorspace
        assert roughness_colorspace in ['auto', 'raw']
        self.roughness_colorspace = roughness_colorspace
        assert metallic_colorspace in ['auto', 'raw']
        self.metallic_colorspace = metallic_colorspace
        assert clearcoat_colorspace in ['auto', 'raw']
        self.clearcoat_colorspace = clearcoat_colorspace
        assert clearcoat_roughness_colorspace in ['auto', 'raw']
        self.clearcoat_roughness_colorspace = clearcoat_roughness_colorspace
        assert opacity_colorspace in ['auto', 'raw']
        self.opacity_colorspace = opacity_colorspace
        assert ior_colorspace in ['auto', 'raw']
        self.ior_colorspace = ior_colorspace
        assert specular_colorspace in ['auto', 'raw', 'sRGB']
        self.specular_colorspace = specular_colorspace
        assert normals_colorspace in ['auto', 'raw', 'sRGB']
        self.normals_colorspace = normals_colorspace
        self.displacement_colorspace = displacement_colorspace
        assert transmittance_colorspace in ['auto', 'raw']
        self.transmittance_colorspace = transmittance_colorspace
        self.is_specular_workflow = is_specular_workflow

    def write_to_usd(self, file_path, scene_path, bound_prims=None, time=None,
                     texture_dir='', texture_file_prefix='', shader='UsdPreviewSurface'):
        raise DeprecationWarning('PBRMaterial.write_to_usd is deprecated; instead use kaolin.io.usd.export_material')

    def read_from_usd(self, file_path, scene_path, texture_path=None, time=None):
        raise DeprecationWarning('PBRMaterial.read_from_usd is deprecated; instead use kaolin.io.usd.import_material')

    def get_attributes(self, only_tensors=False):
        r"""Returns names of all attributes that are currently set.

        Return:
           (list): list of string names
        """
        res = []
        options = (PBRMaterial.__value_attributes + PBRMaterial.__texture_attributes) if only_tensors else PBRMaterial.__slots__
        for attr in options:
            if getattr(self, attr) is not None:
                res.append(attr)
        return res

    def _construct_apply(self, func, attributes=None):
        r"""Creates a shallow copy of self, applies func() to all (or specified) tensor attributes in the copy,
        for example converting to cuda.
        """
        if attributes is None:
            attributes = self.get_attributes(only_tensors=True)

        my_copy = copy.copy(self)
        for attr in attributes:
            current_val = getattr(my_copy, attr)
            if current_val is not None:
                updated_val = func(current_val)
                setattr(my_copy, attr, updated_val)
        return my_copy

    def to(self, device):
        """Returns a copy where all material attributes that are tensors are put on the provided device.
        Note that behavior of member tensors is consistent with PyTorch ``Tensor.to`` method.

        Arguments:
            device (torch.device): The destination GPU/CPU device.

         Returns:
            (PBRMaterial): The new material.
        """
        return self._construct_apply(lambda t: t.to(device=device))

    def cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        """Returns a copy where all the attributes are on CUDA memory.

        Arguments:
            device (torch.device): The destination GPU device. Defaults to the current CUDA device.
            non_blocking (bool): If True and the source is in pinned memory,
                                 the copy will be asynchronous with respect to the host.
            memory_format (torch.memory_format, optional): the desired memory format of returned Tensor.
                                                           Default: torch.preserve_format.

        Returns:
            (PBRMaterial): The new material.
        """
        return self._construct_apply(
            lambda t: t.cuda(device=device, non_blocking=non_blocking, memory_format=memory_format))

    def cpu(self, memory_format=torch.preserve_format):
        """Returns a copy where all the attributes are on CPU memory.

        Arguments:
            memory_format (torch.memory_format, optional): the desired memory format of returned Tensor.
                                                           Default: torch.preserve_format.

        Returns:
            (PBRMaterial): The new material.
        """
        return self._construct_apply(lambda t: t.cpu(memory_format=memory_format))

    def contiguous(self, memory_format=torch.contiguous_format):
        """Returns a copy where all the attributes are contiguous in memory.

        Arguments:
            memory_format (torch.memory_format, optional): the desired memory format of returned Tensor.
                                                           Default: torch.contiguous_format.

        Returns:
            (PBRMaterial): The new material.
        """
        return self._construct_apply(lambda t: t.contiguous(memory_format=memory_format))

    def hwc(self):
        """Returns a copy where all the image attributes are in HWC layout.

        Returns:
            (PBRMaterial): The new material.
        """
        def _to_hwc(val):
            if val.shape[0] in [1, 3, 4]:
                return val.permute(1, 2, 0)
            return val

        return self._construct_apply(lambda t: _to_hwc(t), PBRMaterial.__texture_attributes)

    def chw(self):
        """Returns a copy where all the image attributes are in CHW layout.

        Returns:
            (PBRMaterial): The new material.
        """
        def _to_chw(val):
            if val.shape[2] in [1, 3, 4]:
                return val.permute(2, 0, 1)
            return val

        return self._construct_apply(lambda t: _to_chw(t), PBRMaterial.__texture_attributes)

    def describe_attribute(self, attr, print_stats=False, detailed=False):
        r"""Outputs an informative string about an attribute; the same method
        used for all attributes in ``to_string``.

         Args:
            print_stats (bool): if to print statistics about values in each tensor
            detailed (bool): if to include additional information about each tensor

        Return:
            (str): multi-line string with attribute information
        """
        assert attr in PBRMaterial.__slots__, f"Unsupported attribute {attr}"
        val = getattr(self, attr)
        res = ''
        if attr in PBRMaterial.__value_attributes:
            res = kaolin.utils.testing.tensor_info(
                val, name=f'{attr : >33}', print_stats=print_stats, detailed=detailed) + f' {val}'
        elif torch.is_tensor(val):
            res = kaolin.utils.testing.tensor_info(
                val, name=f'{attr : >33}', print_stats=print_stats, detailed=detailed)
        elif attr in PBRMaterial.__colorspace_attributes:
            if val != "auto":
                res = '{: >33}: {}'.format(attr, val)
        elif val:
            res = '{: >33}: {}'.format(attr, val)
        return res

    def to_string(self, print_stats=False, detailed=False):
        r"""Returns information about attributes as a multi-line string.

        Args:
            print_stats (bool): if to print statistics about values in each tensor
            detailed (bool): if to include additional information about each tensor

        Return:
            (str): multi-line string with attribute information
        """

        res = [f'PBRMaterial object with']
        res.append(self.describe_attribute('material_name'))
        for attr in PBRMaterial.__misc_attributes:
            if attr != 'material_name':
                res.append(self.describe_attribute(attr))
        attributes = self.get_attributes(only_tensors=True)
        for attr in attributes:
            res.append(self.describe_attribute(attr, print_stats=print_stats, detailed=detailed))
        for attr in PBRMaterial.__colorspace_attributes:
            res.append(self.describe_attribute(attr))

        res = [x for x in res if len(x) > 0]
        return '\n'.join(res)

    def __str__(self):
        return self.to_string()


# Useful for testing
def random_material_values(device=None):
    res = {
        'diffuse_color': (random.random(), random.random(), random.random()),
        'roughness_value': random.random(),
        'metallic_value': random.random(),
        'clearcoat_value': random.random(),
        'clearcoat_roughness_value': random.random(),
        'opacity_value': random.random(),
        'opacity_threshold': random.random(),
        'ior_value': random.random(),
        'specular_color':  (random.random(), random.random(), random.random()),
        'displacement_value': random.random(),
        'transmittance_value': random.random(),
        'is_specular_workflow': True,
    }
    if device is not None:
        for k in res.keys():
            res[k] = _to_1d_tensor(res[k]).to(device)
    return res


def random_material_textures(device=None):
    res = {
        'diffuse_texture': torch.rand((3256, 256, 3)),
        'roughness_texture': torch.rand((256, 256, 1)),
        'metallic_texture': torch.rand((256, 256, 1)),
        'clearcoat_texture': torch.rand((256, 256, 1)),
        'clearcoat_roughness_texture': torch.rand((256, 256, 1)),
        'opacity_texture': torch.rand((256, 256, 1)),
        'ior_texture': torch.rand((256, 256, 1)),
        'specular_texture': torch.rand((256, 256, 3)),
        'normals_texture': torch.rand((256, 256, 1)),
        'displacement_texture': torch.rand((256, 256, 3)),
        'transmittance_texture': torch.rand((256, 256, 1)),
    }
    if device is not None:
        for k in res.keys():
            res[k] = res[k].to(device)
    return res


def random_material_colorspaces():
    return {'diffuse_colorspace': ['auto', 'raw'][random.randint(0, 1)],
            'roughness_colorspace': ['auto', 'raw'][random.randint(0, 1)],
            'metallic_colorspace': ['auto', 'raw'][random.randint(0, 1)],
            'clearcoat_colorspace': ['auto', 'raw'][random.randint(0, 1)],
            'clearcoat_roughness_colorspace': ['auto', 'raw'][random.randint(0, 1)],
            'opacity_colorspace': ['auto', 'raw'][random.randint(0, 1)],
            'ior_colorspace': ['auto', 'raw'][random.randint(0, 1)],
            'specular_colorspace': ['auto', 'raw'][random.randint(0, 1)],
            'normals_colorspace': ['auto', 'raw'][random.randint(0, 1)],
            'displacement_colorspace': ['auto', 'raw'][random.randint(0, 1)],
            'transmittance_colorspace': ['auto', 'raw'][random.randint(0, 1)]}