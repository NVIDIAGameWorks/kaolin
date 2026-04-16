# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

from __future__ import annotations

import logging
import math
import torch
import warnings

import kaolin.utils.testing

from .tensor_container import TensorContainerBase

logger = logging.getLogger(__name__)

def _safe_cat_with_message(parts, skip_errors, message):
    result = None
    try:
        result = torch.cat(parts, dim=0)
    except Exception as e:
        if skip_errors:
            logger.error(f'{message}, skipping: {e}')
        else:
            raise ValueError(f'{message}: {e}')
    return result


class PointSamples(TensorContainerBase):
    r"""Base container for point-based 3D representations built over PyTorch tensors.

    Stores ``positions`` of shape ``(N, 3)`` and optional ``features`` as a tensor of
    any dimensionality with shape ``(N, ...)`` or a ``dict`` of tensors ``(N, ...)``, where
    different tensors can have any different dimensionalities, so long as the first dimension
    is ``N``. This class is implemented in a general way, such that **subclasses inherit
    useful point-based utilities** even when adding extra attributes.

    .. rubric:: General utility methods

    These are inherited from :class:`TensorContainerBase`, but still work on new attributes.

    * :meth:`to` - move tensor attributes to ``device`` or ``dtype``
    * :meth:`cuda`, :meth:`cpu` - move tensor attributes to cuda/CPU devices.
    * :meth:`detach` - detach all tensor attributes
    * :meth:`to_string(print_stats=True)` - easy inspection, also allows ``print(obj)`` to work
    * :meth:`as_dict()` - saves all attributes to dict, compatible with constructor ``PointSamples(**dict_output)``
    * :meth:`get_attributes` - return all non-`None` attribute names
    * :meth:`assert_supported` - returns True if attribute name is supported
    * :meth:`check_sanity` - checks all tensors for sanity

    .. rubric:: Point-specific utility methods

    These are point-specific utilities, and will be inherited by subclasses.

    * :meth:`cat` - concatenate all tensors along the point dimension, including features (override :meth:`_custom_attr_cat` to customize)
    * :meth:`as_transformed` - apply affinetransform to positions (override to customize)
    * :meth:`\[mask\] <__getitem__>` - return new instance with point properties masked by boolean mask
    * :meth:`len <__len__>` - return number of points

    .. rubric:: Inheriting from PointSamples

    Inherit from ``PointSamples`` whenever you want to manage any number of attributes
    associated to points and want to define custom behavior for some of the attributes.
    To inherit, simply define (e.g. see :class:`GaussianSplatModel`):

    .. code-block:: python

        class MyAugmentedPoints(PointSamples):
            @classmethod
            def class_tensor_attributes(cls):
                return ["positions", "transform", "features"] + ... custom attributes

            @classmethod
            def class_other_attributes(cls):
                return [] + ... custom non-tensor attributes

            @classmethod
            def class_point_attributes(cls):
                return ["positions", "features"] + ... custom per-point attributes (subset of tensor attributes)

            def check_tensor_attribute_shape(self, attr):
                # check if geattr(self, attr) has expected shape and return True if it does
                pass

            # Optional ----------------------------------
            def _custom_attr_cat(cls, models, attr, skip_errors=False, **kwargs):
                # Define custom concatenation behavior for any attr; return None for default

            def as_transformed(self, transform=None):
                # Define custom transform behavior

    """
    @classmethod
    def class_tensor_attributes(cls):
        """Class attribute names that are PyTorch tensors."""
        return ["positions", "transform", "features"]

    @classmethod
    def class_other_attributes(cls):
        """Class attribute names that are not PyTorch tensors."""
        return []

    @classmethod
    def class_point_attributes(cls):
        """Subset of class tensor attributes that contain per-point values, so sized as (N,...), where N is num points."""
        return ["positions", "features"]

    def __init__(self, positions, features=None, transform=None, strict_checks: bool = True):
        """ Initializes the class and optionally validates the inputs.

        Args:
            positions (torch.Tensor): Point positions, shape ``(N, 3)``.
            features (torch.Tensor or dict, optional): Per-point features as a tensor of any dimensionality
                with first dimension equal to ``N``, i.e. ``(N, ...)``, or a dict of feature tensors of any varying
                dimensionalities, as long as the first dimension is ``N``, e.g. ``{"a": (N, F_0), "b": (N, F_1, F_2)}``.
            transform (torch.Tensor, optional): Global affine transform of shapes ``(1, 4, 4)`` or ``(4, 4)``, or
                per-point transforms ``(N, 4, 4)``.
            strict_checks (bool): If ``True``, validate tensor shapes on construction and raise error.

        Raises:
            ValueError: if `strict_checks` is True and the inputs are invalid.
        """
        self.positions = positions
        self.features = features
        self.transform = transform

        if strict_checks:
            if not self.check_sanity():
                raise ValueError(f'Illegal inputs passed to {self.__class__.__name__} constructor; check log')

    def check_tensor_attribute_shape(self, attr):
        """Checks that the tensor stored in ``attr`` has an expected shape.

        Per-attribute shape validation hook used by :meth:`_check_tensor_attribute`.
        Override in subclasses to add support for custom tensor attributes.

        Args:
            attr (str): attribute name (must be in :meth:`class_tensor_attributes`).

        Returns:
            bool: ``True`` if the tensor at ``attr`` has the expected shape, ``False`` otherwise.

        Raises:
            ValueError: If ``attr`` is not a tensor attribute supported by this class.
        """
        value = getattr(self, attr)
        if attr == 'positions':
            return kaolin.utils.testing.check_tensor(value, shape=(len(self), 3), throw=False)
        elif attr == 'transform':
            return kaolin.utils.testing.check_tensor(value, shape=(len(self), 4, 4), throw=False) or \
                kaolin.utils.testing.check_tensor(value, shape=(1, 4, 4), throw=False) or \
                kaolin.utils.testing.check_tensor(value, shape=(4, 4), throw=False)
        else:
            raise ValueError(f'check_tensor_attribute_shape not implemented for {attr}')

    def check_tensor_attribute(self, attr, log_error=False):
        """Checks tensor attribute validity; returns True if valid."""
        if attr == 'features':
            return self._check_features_attribute(log_error)
        elif attr == 'transform':
            if getattr(self, 'transform') is None:
                return True
            else:
                return self._check_tensor_attribute(attr, log_error)
        else:
            return self._check_tensor_attribute(attr, log_error)

    def _check_tensor_attribute(self, attr, log_error=True):
        def _maybe_log(msg):
            if log_error:
                logger.error(msg)

        # Note: right now, all attributes except features are treated as required
        value = getattr(self, attr)
        if value is None:
            _maybe_log(f'Attribute {attr} is None, but must be set')
            return False

        if not torch.is_tensor(value):
            _maybe_log(f'Attribute {attr} must be torch.Tensor, but is {type(value)}')
            return False

        if not self.check_tensor_attribute_shape(attr):
            _maybe_log(f'Attribute {self.describe_attribute(attr)} does not match expected shape')
            return False

        # in case subclass did not implement expected_shape, we still check point attributes
        if attr in self.class_point_attributes():
            if value.shape[0] != len(self):
                _maybe_log(f'Point-level attribute {attr} has shape {value.shape}, but {len(self)} expected for dim=0')
                return False
        return True

    def _check_features_attribute(self, log_error=True):
        def _maybe_log(msg):
            if log_error:
                logger.error(msg)

        value = self.features
        if value is None:
            return True

        def _check_tensor_val(val, name):
            if not torch.is_tensor(val):
                _maybe_log(f'Unexpected type {type(value)} for {name}')
                return False
            if val.shape[0] != len(self):
                _maybe_log(f'Expected one feature per point, but {name} has shape {val.shape}')
                return False
            return True

        if torch.is_tensor(value):
            if not _check_tensor_val(value, 'features'):
                return False
        elif isinstance(value, dict):
            for k, v in value.items():
                if not _check_tensor_val(v, f'features[{k}]'):
                    return False
        else:
            _maybe_log(f'Features have unexpected type {type(value)} (dict or tensor expected)')
            return False
        return True

    def __len__(self):
        """Returns the number of points."""
        return self.positions.shape[0]

    def describe_attribute(self, attr, print_stats=False, detailed=False):
        r"""Outputs an informative string about an attribute; the same method
        used for all attributes in ``to_string``.

        Args:
            attr (str): attribute name
            print_stats (bool): if to print statistics about values in each tensor
            detailed (bool): if to include additional information about each tensor

        Return:
            (str): multi-line string with attribute information

        Raises:
            ValueError if attr is not supported
        """
        self.assert_supported(attr)

        val = super().__getattribute__(attr)

        # TODO: add this XYZ stat to meshes too
        if attr == 'positions':
            res = kaolin.utils.testing.tensor_info(
                val, name=f'{attr : >20}', print_stats=False, detailed=detailed)
            if print_stats:
                res = [res]
                for i, n in enumerate(['x', 'y', 'z']):
                    res.append(f'{n : >25} - [min %0.4f, max %0.4f, mean %0.4f]' % \
                       (torch.min(val[:, i]).item(),
                        torch.max(val[:, i]).item(),
                        torch.mean(val[:, i].to(torch.float32)).item()))
                res = '\n'.join(res)
            return res
        else:
            return super().describe_attribute(attr, print_stats=print_stats, detailed=detailed)

    def _to_string_class_summary(self):
        return f'{self.__class__.__name__} of {len(self)}'

    @classmethod
    def _custom_attr_cat(cls, models, attr, skip_errors=False, **kwargs):
        """Override in subclasses to provide custom concatenation for a given attribute.

        Return a non-``None`` value to short-circuit the default logic in
        :meth:`_attr_cat`; return ``None`` to fall through to default handling.
        """
        return None

    @classmethod
    def _attr_cat(cls, models, attr, skip_errors=False, **kwargs):
        """Concatenate attribute ``attr`` across a list of models.

        Calls :meth:`_custom_attr_cat` first; falls back to default tensor/other-attribute
        concatenation logic if that returns ``None``.

        Args:
            models (list): Instances of this class to concatenate.
            attr (str): Attribute name to concatenate.
            skip_errors (bool): If ``True``, log and skip mismatches instead of raising.

        Returns:
            Concatenated value, or ``None`` if all inputs were ``None``.
        """
        result = cls._custom_attr_cat(models, attr, skip_errors, **kwargs)
        if result is not None:
            return result

        if attr in cls.class_tensor_attributes():
            parts = [getattr(m, attr) for m in models]

            if all(p is None for p in parts):
                pass
            elif any(p is None for p in parts):
                if skip_errors:
                    logger.error(f'Attribute {attr} only present on some input objects, skipping')
                else:
                    raise ValueError(
                        f'Attribute {attr} must be set on all models or None on all models to cat')
            else:
                if attr == "features":
                    if not all(type(x) is type(parts[0]) for x in parts):
                        if skip_errors:
                            logger.error(f'Attribute "features" has different types on some input objects, skipping')
                        else:
                            raise ValueError(
                                f'Some instance features have different types {[type(x) for x in parts]}')
                    elif torch.is_tensor(parts[0]):
                        result = _safe_cat_with_message(
                            parts, skip_errors, f'Failed to cat "features" attribute with error')
                    elif not isinstance(parts[0], dict):
                        if skip_errors:
                            logger.error(f'Features have unexpected type {type(parts[0])} (dict or tensor expected), skipping')
                        else:
                            raise ValueError(
                                f'Features can only be tensor or dict, found {type(parts[0])}')
                    else:
                        keys = parts[0].keys()
                        common_keys = set(keys)
                        for p in parts[1:]:
                            if set(p.keys()) != set(keys):
                                if skip_errors:
                                    common_keys = common_keys.intersection(set(p.keys()))
                                else:
                                    raise ValueError(
                                        f'Instances have mismatched feature keys: {list(keys)} vs {list(p.keys())}')

                        new_value = {}
                        for k in common_keys:
                            k_value = _safe_cat_with_message(
                                [p[k] for p in parts], skip_errors, f'Failed to cat features[{k}]')
                            if k_value is not None:
                                new_value[k] = k_value
                        if len(new_value) > 0:
                            result = new_value
                else:  # not "features"
                    result = _safe_cat_with_message(
                        parts, skip_errors, f'Failed to cat attribute {attr}')

        elif attr in cls.class_other_attributes():
            vals = [getattr(m, attr) for m in models]
            if not all(v == vals[0] for v in vals[1:]):
                warnings.warn(
                    f'Attribute {attr} has different values across models; '
                    f'using value from first model')
            result = vals[0]

        return result

    @classmethod
    def cat(cls, models, skip_errors=False, **kwargs):
        """Concatenates a list of instances along the point dimension.

        Any stored ``transform`` on each model is applied before concatenation;
        the result always has ``transform=None``.

        Args:
            models (list): Non-empty list of instances of this class.
            skip_errors (bool): If ``True``, log and skip mismatched attributes
                instead of raising.

        Returns:
            New instance with all point attributes concatenated.
        """
        # Apply any stored transforms before concatenation; result will have transform=None
        models = [m.as_transformed() if m.transform is not None else m for m in models]

        if len(models) == 0:
            raise ValueError(
                'Zero length list provided to cat operation; at least 1 model input required')

        if len(models) == 1:
            return models[0]

        res_kwargs = {}
        for attr in cls.class_tensor_attributes() + cls.class_other_attributes():
            res_kwargs[attr] = cls._attr_cat(models, attr, skip_errors=skip_errors, **kwargs)

        return cls(**res_kwargs)

    # TODO: consider extending to slices, etc.
    def __getitem__(self, mask):
        """Return a new instance with ``mask`` applied to all per-point attributes.

        Args:
            mask (torch.Tensor): Boolean tensor of shape ``(N,)``.

        Returns:
            PointSamples: New instance of the same class containing only selected points.
        """
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f'Mask must be a torch.Tensor, got {type(mask)}')
        if mask.dtype != torch.bool:
            raise TypeError(f'Mask must be boolean, got {mask.dtype}')
        if mask.shape[0] != len(self):
            raise ValueError(f'Mask length {mask.shape[0]} does not match number of points {len(self)}')

        kwargs = {}
        for attr in self.get_attributes():
            if attr in self.class_point_attributes():
                val = getattr(self, attr)
                if isinstance(val, dict):
                    kwargs[attr] = {k: v[mask] for k, v in val.items()}
                else:
                    kwargs[attr] = val[mask]
            elif attr in self.class_tensor_attributes() or attr in self.class_other_attributes():
                kwargs[attr] = getattr(self, attr)
        return self.__class__(**kwargs)

    def _combined_canonical_transform(self, input_transform=None):
        left_transform = input_transform
        right_transform = self.transform

        if left_transform is None and right_transform is None:
            return None

        if left_transform is None:
            left_transform = torch.eye(4, device=right_transform.device, dtype=right_transform.dtype).unsqueeze(0)
        elif len(left_transform.shape) == 2:
            left_transform = left_transform.unsqueeze(0)
        if right_transform is None:
            right_transform = torch.eye(4, device=left_transform.device, dtype=left_transform.dtype).unsqueeze(0)
        elif len(right_transform.shape) == 2:
            right_transform = right_transform.unsqueeze(0)

        final_transform = left_transform @ right_transform
        return final_transform
        

    def as_transformed(self, additional_transform=None):
        """Uses stored transform (if set) or `additional_transform`, or chains `additional_transform @ self.transform`
        if both are set, and returns a new instance of the class, with the transform applied. Works for any
        affine transform.

        Args:
            additional_transform (optional, torch.Tensor): if not set, will use transform set on the class; should
                be affine transform of shape (4,4) or (1,4,4) or (N,4,4) where N is the number of points.

        .. note::
           This method does not copy all the attributes, only transformable ones result in new tensors;
           if full copy is required, first call copy.deepcopy(). Only isotropic scale, rotation and translation
           can be applied consistently.

        Returns:
            PointSamples: new instance of :class:`PointSamples` with the transform applied.
        """
        # Compose: if both argument and stored transform are set, chain them
        transform = self._combined_canonical_transform(additional_transform)

        res = self.as_dict()
        if transform is not None:
            res['positions'] = (transform[..., :3, :3] @ self.positions[:, :, None] + transform[..., :3, 3:]).squeeze(-1)
            res['transform'] = None
        return self.__class__(**res)


class GaussianSplatModel(PointSamples):
    r"""Container for a **3D Gaussian Splat** cloud of ``N`` splats.
    Extends :class:`PointSamples` with Gaussian-specific attributes, inheriting
    generalized tensor and point-level utilities, summarized below.

    .. rubric:: Supported Attributes:

    ``GaussianSplatModel`` supports the following attributes.

    .. list-table::
       :header-rows: 1
       :widths: 20 25 55

       * - **Attribute**
         - **Shape**
         - **Description**
       * - ``positions``
         - ``(N, 3)``
         - Splat centres
       * - ``orientations``
         - ``(N, 4)``
         - Unit quaternions :math:`(w,x,y,z)`
       * - ``scales``
         - ``(N, 3)``
         - Per-axis scale, post activation
       * - ``opacities``
         - ``(N,)``
         - Opacity per splat, post activation
       * - ``sh_coeff``
         - ``(N, S, 3)``
         - SH coefficients; :math:`S = (sh\_degree + 1)^2`
       * - ``features``
         - ``(N, ...)`` or dict
         - Per-point features *(optional)*
       * - ``transform``
         - ``(4, 4)`` or ``(N, 4, 4)`` or ``(1, 4, 4)`` or ``None``
         - Affine transform *(optional)*, stored, not applied

           (see :meth:`as_transformed`)
       * - ``sh_degree``
         - *int*
         - SH degree :math:`L`; inferred from ``sh_coeff`` if omitted


    .. rubric:: General utility methods

    These are inherited, but still work on all attributes, including tensor of dict of tensors for `features`.

    * :meth:`to` - move tensor attributes to ``device`` or ``dtype``
    * :meth:`cuda`, :meth:`cpu` - move tensor attributes to cuda/CPU devices.
    * :meth:`detach` - detach all tensor attributes
    * :meth:`to_string(print_stats=True)` - easy inspection, also allows ``print(obj)`` to work
    * :meth:`as_dict()` - saves all attributes to dict
      * compatible with constructor ``GaussianSplatModel(**dict_output)``
      * compatible with :func:`~kaolin.io.usd.export_gaussiancloud`
      * compatible with :func:`~kaolin.io.ply.export_gaussiancloud`
    * :meth:`check_sanity` - checks all tensor shapes for sanity
    * :meth:`len <__len__>` - return number of gaussians

    .. rubric:: Gaussian-specific utility methods

    Gaussian-specific utility methods.

    * :meth:`cat` - concatenate all tensors along the point dimension, including features (override :meth:`_custom_attr_cat` to customize)
    * :meth:`as_transformed` - apply affine transform to all gaussian attributes
      * **Note**: ⚠️ Works for only isotropic scaling, rotation, translation, and unexpected results may occur for other transform combinations.
    * :meth:`\[mask\] <__getitem__>` - return new instance with point properties masked by boolean mask
    * :meth:`compute_sh_degree` - compute SH degree based on number of SH coefficients (class method)
    * :meth:`compute_num_sh_coeff` - compute number of SH coefficients based on SH degree (class method)
    """

    @classmethod
    def class_tensor_attributes(cls):
        """Class attribute names that are PyTorch tensors."""
        return ['positions', 'orientations', 'scales', 'opacities', 'sh_coeff', 'features', 'transform']

    @classmethod
    def class_other_attributes(cls):
        """Class attribute names that are not PyTorch tensors."""
        return ["sh_degree"]

    @classmethod
    def class_point_attributes(cls):
        """Subset of class tensor attributes that contain per-point values, so sized as (N,...), where N is num points."""
        return PointSamples.class_point_attributes() + ["orientations", "scales", "opacities", "sh_coeff"]

    def __init__(self, positions, orientations, scales, opacities, sh_coeff, features=None, transform=None, sh_degree=None,
                 strict_checks: bool = True):
        """
        Initializes the class and optionally validates the inputs. 

        👉 Note: all attributes are stored **post-activation** in their final range. Override class to customize.

        Args:
            positions (torch.Tensor): Splat centres, shape ``(N, 3)``.
            orientations (torch.Tensor): Unit quaternions ``(N, 4)``, ``wxyz`` convention (will be normalized internally).
            scales (torch.Tensor): Per-axis scale ``(N, 3)``.
            opacities (torch.Tensor): Opacity per splat, shape ``(N,)``.
            sh_coeff (torch.Tensor): SH coefficients ``(N, S, 3)`` where
                ``S = (sh_degree + 1) ** 2``.
            features (torch.Tensor or dict, optional): Arbitrary per-point features.
            transform (torch.Tensor, optional): Global affine transform of shapes ``(1, 4, 4)`` or ``(4, 4)``, or
                per-point transforms ``(N, 4, 4)``.
            sh_degree (int, optional): SH degree.  If ``None``, inferred from
                ``sh_coeff.shape[1]``.
            strict_checks (bool): If ``True``, validates tensor shapes on construction and raises error if invalid.

        Raises:
            ValueError: if `strict_checks` is True and the inputs are invalid.
        """
        super().__init__(positions, features, transform=transform, strict_checks=False)  # don't apply super sanity checks
        self.orientations = torch.nn.functional.normalize(orientations)
        self.scales = scales
        self.opacities = opacities
        self.sh_coeff = sh_coeff

        if sh_degree is None:
            sh_degree = self.compute_sh_degree(sh_coeff.shape[1])
        self.sh_degree = sh_degree

        if strict_checks:
            if not self.check_sanity():
                raise ValueError(f'Illegal inputs passed to {self.__class__.__name__} constructor; check log')

    def check_other_attribute(self, attr, log_error=False):
        """Performs custom checks for gaussian-specific non-tensor attributes.

        Args:
            attr (str): Attribute name.
            log_error (bool): If ``True``, logs error messages.

        Returns:
            bool: True if attribute is valid, False otherwise.
        """
        if attr == 'sh_degree':
            expected_sh = self.compute_sh_degree(self.sh_coeff.shape[1])
            if self.sh_degree != expected_sh:
                if log_error:
                    logger.error(f'Attribute sh_degree has value {self.sh_degree}, but expected {expected_sh} based on sh_coeff')
                return False
        return True

    def check_tensor_attribute_shape(self, attr):
        """Performs custom shape checks for gaussian-specific tensor attributes.

        Args:
            attr (str): Attribute name.

        Returns:
            bool: True if shape is valid, False otherwise.
        """
        self.assert_supported(attr)
        n = len(self)
        expected_shapes = {
            'orientations': (n, 4),
            'scales': (n, 3),
            'opacities': (n,),
            'sh_coeff': (n, self.compute_num_sh_coeff(self.sh_degree), 3)
        }
        if attr in expected_shapes:
            value = getattr(self, attr)
            return kaolin.utils.testing.check_tensor(value, shape=expected_shapes[attr], throw=False)
        else:
            return super().check_tensor_attribute_shape(attr)

    @classmethod
    def compute_sh_degree(cls, num_sh_coeff):
        """Computes SH degree based on *total* number of SH coefficients, i.e. second dim of ``sh_coeff``.

        Returns:
            int: SH degree.

        Raises:
            ValueError: If ``num_sh_coeff`` is not a perfect square.
        """
        iroot = math.isqrt(num_sh_coeff)
        root = math.sqrt(num_sh_coeff)
        if abs(iroot - root) > 0.001:
            raise ValueError(f'Num SH coefficients {num_sh_coeff} is not a square of an int; failing to compute sh_degree')
        sh_guess = int(iroot) - 1
        return sh_guess

    @classmethod
    def compute_num_sh_coeff(cls, sh_degree):
        """Computes expected number of total sh_coeff features (i.e. 2nd dim) based on sh_degree.

        Returns:
            int: Number of SH coefficients.
        """
        expected_val = (sh_degree + 1) ** 2
        return expected_val

    def _to_string_class_summary(self):
        return f'{self.__class__.__name__} of {len(self)} (sh_degree={self.sh_degree})'

    @classmethod
    def _custom_attr_cat(cls, models, attr, skip_errors=False, **kwargs):
        """ Provides special handling of concatenating sh coefficients, capping at min among the list of models."""
        # TODO: add flexibility by supporting sh_mode='min'|'max'|'match'
        if attr == 'sh_degree':
            return min(m.sh_degree for m in models)
        elif attr == 'sh_coeff':
            nums = [m.sh_coeff.shape[1] for m in models]
            if not all(x == nums[0] for x in nums):
                if skip_errors:
                    logger.warning(f'Number of SH coefficients varies; capping to lowest value')
                    min_shape = min(nums)
                    return _safe_cat_with_message(
                        [m.sh_coeff[:, :min_shape, :] for m in models], skip_errors,
                        f'Failed to cat sh_coeff capped to {min_shape} features, skipping')
                else:
                    raise ValueError(f'Mismatch between number of features in "sh_coeff" attribute')
        return None  # default handling

    def as_transformed(self, additional_transform=None):
        """Uses stored transform (if set) or `additional_transform`, or chains `additional_transform @ self.transform`
        if both are set, and returns a new instance of the class, with the transform applied.

        **Note**: ⚠️ For Gaussians, works robustly for isotropic scaling, rotation, translation (combined in 
        a general transform matrix), but unexpected results may occur for shear and anisotropic scaling (for 
        example, applying inverse transform will not work correctly for these cases).

        Args:
            additional_transform (optional, torch.Tensor): if not set, will use transform set on the class; should
                be affine transform of shape (4,4) or (1,4,4) or (N,4,4) where N is the number of Gaussians.

        .. note::
           This method does not copy all the attributes, only transformable ones result in new tensors;
           if full copy is required, first call copy.deepcopy(). Only isotropic scale, rotation and translation
           can be applied consistently.

        Returns:
            GaussianSplatModel: new instance of :class:`GaussianSplatModel` with the transform applied.
        """
        from kaolin.ops.gaussians import transform_gaussians

        transform = self._combined_canonical_transform(additional_transform)

        kwargs = self.as_dict()
        if transform is not None:
            kwargs['positions'], kwargs['orientations'], kwargs['scales'], kwargs['sh_coeff'][:, 1:, :] = transform_gaussians(
                self.positions, self.orientations, self.scales, transform, shs_feat=self.sh_coeff[:, 1:, :],
                use_log_scales=False)
            kwargs['transform'] = None
        return GaussianSplatModel(**kwargs)




