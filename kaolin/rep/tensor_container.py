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

from abc import ABC, abstractmethod
import copy
from typing import Any, Dict, Optional, Sequence, Union

import torch

import kaolin.utils.testing


class TensorContainerBase(ABC):
    r"""Abstract base class for dealing with containers of tensors.

    Subclasses describe their attributes by implementing
    :meth:`class_tensor_attributes`, :meth:`class_other_attributes`, and
    :meth:`check_tensor_attribute`; in exchange they get many useful utilities
    out of the box (see below). Tensor attributes may either be a single
    :class:`torch.Tensor` or a ``dict`` of tensors.

    .. rubric:: General utility methods

    These are inherited and operate on any subclass without further work.

    * :meth:`to` - move tensor attributes to ``device`` or ``dtype``
    * :meth:`cuda`, :meth:`cpu` - move tensor attributes to cuda/CPU devices
    * :meth:`detach` - detach all tensor attributes
    * :meth:`to_string(print_stats=True)` - easy inspection, also allows ``print(obj)`` to work
    * :meth:`as_dict` - saves all attributes to dict, should be compatible with constructor
    * :meth:`get_attributes` - return all non-``None`` attribute names
    * :meth:`assert_supported` - raises if attribute name is not supported
    * :meth:`check_sanity` - checks all tensors for sanity (using subclass hooks)

    .. rubric:: Inheriting from TensorContainerBase

    Inherit from ``TensorContainerBase`` whenever you want to manage any number of
    tensor attributes (and optionally non-tensor attributes) and reuse the utilities
    above. To inherit, simply define (e.g. see :class:`~kaolin.rep.PointSamples`):

    .. code-block:: python

        import torch
        from kaolin.rep import TensorContainerBase

        class MyContainer(TensorContainerBase):
            @classmethod
            def class_tensor_attributes(cls):
                return ["data", "extras"]

            @classmethod
            def class_other_attributes(cls):
                return ["label"]

            def __init__(self, data, extras=None, label="default"):
                self.data = data
                self.extras = extras
                self.label = label

            def check_tensor_attribute(self, attr, log_error=False):
                value = getattr(self, attr)
                if attr == "data":
                    return torch.is_tensor(value)
                if attr == "extras":
                    return value is None or (
                        isinstance(value, dict) and all(torch.is_tensor(v) for v in value.values()))
                return False

            # Optional: override to validate non-tensor attributes
            def check_other_attribute(self, attr, log_error=False):
                if attr == "label":
                    return isinstance(self.label, str)
                return False

        instance = MyContainer(data=torch.randn(4, 3), extras={"a": torch.randn(4, 2)})
        print(instance)                          # uses to_string()
        moved = instance.to(dtype=torch.float64) # all tensor attributes converted
        assert instance.check_sanity()
    """
    @classmethod
    @abstractmethod
    def class_tensor_attributes(cls):
        """Returns attribute names that are PyTorch tensors or dicts thereof."""
        pass

    @classmethod
    @abstractmethod
    def class_other_attributes(cls):
        """Returns attribute names that are not PyTorch tensors or dicts thereof."""
        pass

    @abstractmethod
    def check_tensor_attribute(self, attr, log_error=False):
        """Checks tensor attribute validity; returns True if ok."""
        pass

    def check_other_attribute(self, attr, log_error=False):
        """Checks a non-tensor attribute validity; returns True if ok."""
        other_attributes = self.class_other_attributes()
        if len(other_attributes) > 0:
            raise NotImplementedError(f'Class {self.__class__.__name__} should override check_other_attribute for {other_attributes}')

    @classmethod
    def assert_supported(cls, attr):
        """Raises an exception if class does not support provided attribute name."""
        if attr not in cls.class_tensor_attributes() and attr not in cls.class_other_attributes():
            raise AttributeError(f'{cls.__name__} does not support attribute named "{attr}"')

    def _construct_apply(self, func, attributes=None):
        """Shallow copy; apply ``func`` to selected or all non-``None`` tensor attributes."""
        if attributes is None:
            attributes = self.get_attributes(only_tensors=True)
        my_copy = copy.copy(self)
        for attr in attributes:
            current = getattr(my_copy, attr)
            if isinstance(current, dict):
                setattr(my_copy, attr, {k: func(v) for k, v in current.items()})
            else:
                setattr(my_copy, attr, func(current))
        return my_copy

    def to(self, *args: Any, attributes: Optional[Sequence[str]] = None, **kwargs: Any):
        """Moves or casts tensors like :meth:`torch.Tensor.to` / :meth:`torch.nn.Module.to`.

        Args:
            *args: forwarded to ``tensor.to(*args)``
            attributes (list of str, optional): if set, only these tensor attributes are updated
            **kwargs: forwarded to ``tensor.to(**kwargs)``

        Returns:
            PointSamples: shallow copy with converted tensors
        """
        return self._construct_apply(lambda t: t.to(*args, **kwargs), attributes)

    def cuda(self, device: Optional[Union[int, torch.device, str]] = None,
             attributes: Optional[Sequence[str]] = None):
        """Calls ``cuda`` on all or selected tensor attributes; returns a shallow copy."""
        return self._construct_apply(lambda t: t.cuda(device), attributes)

    def cpu(self, attributes: Optional[Sequence[str]] = None):
        """Calls ``cpu`` on all or selected tensor attributes; returns a shallow copy."""
        return self._construct_apply(lambda t: t.cpu(), attributes)

    def detach(self, attributes: Optional[Sequence[str]] = None):
        """Detaches all or selected tensor attributes; returns a shallow copy."""
        return self._construct_apply(lambda t: t.detach(), attributes)

    def get_attributes(self, only_tensors=False):
        r"""Returns names of all attributes that are currently set to non-None value in this class instance.

        Args:
            only_tensors: if true, will only include tensor attributes

        Return:
           (list): list of string names
        """
        res = []
        options = self.class_tensor_attributes() + ([] if only_tensors else self.class_other_attributes())
        for attr in options:
            val = getattr(self, attr)
            if val is not None:
                res.append(attr)
        return res

    def as_dict(self, only_tensors=False) -> Dict[str, Any]:
        """Return all non-``None`` attributes as a ``{name: value}`` dict."""
        res = {}
        for attr in self.get_attributes(only_tensors=only_tensors):
            res[attr] = getattr(self, attr)
        return res

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

        if isinstance(val, dict):
            parts = []
            for key, fval in val.items():
                parts.append(kaolin.utils.testing.tensor_info(
                    fval, name=f'{attr}[{key}]', print_stats=print_stats, detailed=detailed))
            res = '\n'.join(parts)
        else:
            res = kaolin.utils.testing.tensor_info(
                val, name=f'{attr : >20}', print_stats=print_stats, detailed=detailed)
        return res

    def check_sanity(self, log_error=True):
        """Validates that all tensor attributes are correct; implement abstract methods.

        Args:
            log_error (bool): If ``True``, logs each failed check via ``logger.error``.

        Returns:
            bool: ``True`` if all checks pass, ``False`` otherwise.
        """
        res = True
        for attr in self.class_tensor_attributes():
            res = res and self.check_tensor_attribute(attr, log_error=log_error)

        for attr in self.class_other_attributes():
            res = res and self.check_other_attribute(attr, log_error=log_error)

        return res

    def _to_string_class_summary(self):
        """String used to describe class, override to customize."""
        return f'{self.__class__.__name__}'

    def to_string(self, print_stats=False, detailed=False):
        r"""Returns information about tensor attributes currently contained in the object.

        Args:
            print_stats (bool): if to print statistics about values in each tensor
            detailed (bool): if to include additional information about each tensor

        Return:
            (str): multi-line string with attribute information
        """
        lines = [self._to_string_class_summary()]

        for attr in self.get_attributes():
            lines.append(self.describe_attribute(attr, print_stats=print_stats, detailed=detailed))

        return '\n'.join(lines)

    def __repr__(self):
        return self.to_string()

    def __str__(self):
        return self.to_string()
