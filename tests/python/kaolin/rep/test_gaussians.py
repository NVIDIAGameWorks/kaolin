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

"""Tests for :class:`kaolin.rep.gaussians.GaussianSplatModel` (skeleton only)."""

import copy
import random
import pytest
import torch
import math
import numpy as np
from kaolin.rep import (PointSamples,
                        GaussianSplatModel)
import kaolin.math.quat

from kaolin.utils.testing import contained_torch_equal, with_seed

devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


class BaseTests:
    """Placeholder for GaussianSplatModel unit tests."""
    ClassTested = None
    class_required_attributes = []
    class_optional_attributes = []

    def sample_feature_shape(self):
        shape_length = random.randint(1, 3)
        return [random.randint(1, 4) for _ in range(shape_length)]

    def sample_features_input(self, num_pts, with_features, device="cpu", dtype=torch.float32, feature_shape=None):
        def _random_feature_vector(fs):
            if fs is None:
                fs = self.sample_feature_shape()
            shape = [num_pts] + fs
            return torch.randn(shape, device=device, dtype=dtype)

        if with_features is None:
            return None
        elif isinstance(with_features, bool):
            return _random_feature_vector(feature_shape)
        elif isinstance(with_features, list):
            return {feature: _random_feature_vector(feature_shape) for feature in with_features}

    @pytest.fixture(params=devices)
    def device(self, request):
        return request.param

    @with_seed(42)
    @pytest.fixture(params=[
        {'axis': torch.tensor([0, 1, 0]), 'angle': math.pi / 2},
        {'axis': torch.tensor([1, 0, 0]), 'angle': random.uniform(0.1, math.pi)},
        {'axis': torch.nn.functional.normalize(torch.randn(3), dim=0), 'angle': random.uniform(0.1, math.pi)}
    ])
    def transform_matrix_info(self, request):
        axis, angle = request.param['axis'], request.param['angle']
        rot_matrix = kaolin.math.quat.rot33_from_angle_axis(torch.tensor(angle).unsqueeze(0),
                                                            axis.unsqueeze(0)).squeeze(0)
        result = torch.eye(4)

        scale = torch.abs(torch.randn(1).repeat(3))
        scale_mat = torch.diag(scale)
        result[:3, :3] = rot_matrix @ scale_mat

        translation = torch.randn(3)
        result[:3, 3] = translation
        return result, scale_mat, rot_matrix

    def sample_input_kwargs(self, num_pts, optional_attributes=None, with_features: bool | list[str] = False, device="cpu", dtype=torch.float32, **kwargs):
        raise NotImplementedError('Implement this test for all test subclasses')

    def expected_tensor_attributes(self):
        raise NotImplementedError('Implement this test for all test subclasses')

    def expected_point_attributes(self):
        raise NotImplementedError('Implement this test for all test subclasses')

    def test_sanity_class_specific(self):
        raise NotImplementedError(f'Implement this test for all test subclasses')

    def attributes_changed_by_transform(self):
        raise NotImplementedError(f'Implement this for all test subclasses')

    def test_class(self):
        cls = self.ClassTested
        assert set(cls.class_point_attributes()).issubset(set(cls.class_tensor_attributes()))
        assert set(cls.class_tensor_attributes()) == set(self.expected_tensor_attributes()), \
            f"{cls.class_tensor_attributes()} for {cls.__name__} does not match {self.expected_tensor_attributes()}"
        assert set(cls.class_point_attributes()) == set(self.expected_point_attributes()), \
            f"{cls.class_point_attributes()} for {cls.__name__} does not match {self.expected_point_attributes()}"

    @with_seed(42)
    @pytest.fixture()
    def valid_constructor_kwargs(self, request, device):
        sample_input_kwargs = [
            self.sample_input_kwargs(10, device=device),
            self.sample_input_kwargs(9, with_features=True, device=device),
            self.sample_input_kwargs(15, with_features=["color", "radius"], device=device),
            self.sample_input_kwargs(12, optional_attributes=[] if len(
                self.class_optional_attributes) == 0 else np.random.choice(self.class_optional_attributes, size=2,
                                                                           replace=False).tolist(), device=device),
            self.sample_input_kwargs(16, with_features=True, optional_attributes=self.class_optional_attributes,
                                     device=device)
        ]
        return sample_input_kwargs

    def test_constructor_valid(self, valid_constructor_kwargs):
        for kwargs in valid_constructor_kwargs:
            instance = self.ClassTested(**kwargs)
            for attr, value in kwargs.items():
                assert contained_torch_equal(getattr(instance, attr), value, approximate=True), f"Attribute {attr} does not match expected value"

    def test_as_dict(self, valid_constructor_kwargs):
        for kwargs in valid_constructor_kwargs:
            instance = self.ClassTested(**kwargs)

            instance_dict = instance.as_dict()
            reconstructed_instance = self.ClassTested(**instance_dict)

            for attr, value in kwargs.items():
                assert contained_torch_equal(
                    instance_dict[attr], value, approximate=True), \
                    f"Attribute {attr} does not match expected value"
                assert contained_torch_equal(
                    getattr(reconstructed_instance, attr), value, approximate=True), \
                    f"Reconstructed attribute {attr} does not match expected value"

            # Also test as_dict
            reconstructed_dict = reconstructed_instance.as_dict()
            assert contained_torch_equal(instance_dict, reconstructed_dict, approximate=True, print_error_context='')


    @with_seed(42)
    def test_constructor_invalid(self):
        num_pts = 10
        valid_kwargs = self.sample_input_kwargs(num_pts)

        with pytest.raises((TypeError, ValueError)):
            self.ClassTested(**{**valid_kwargs, "positions": None})

        for bad_features in ["a string", 42, [1, 2, 3]]:
            with pytest.raises((TypeError, ValueError)):
                self.ClassTested(**{**valid_kwargs, "features": bad_features})

        with pytest.raises((TypeError, ValueError)):
            self.ClassTested(**{**valid_kwargs, "features": torch.randn(num_pts + 5, 3)})

        with pytest.raises((TypeError, ValueError)):
            self.ClassTested(**{**valid_kwargs, "features": {"color": "not_a_tensor"}})

        with pytest.raises((TypeError, ValueError)):
            self.ClassTested(**{**valid_kwargs, "features": {"color": torch.randn(num_pts + 5, 3)}})

        # Let's try messing with shape of other attributes
        for attr in self.ClassTested.class_point_attributes():
            if attr not in ["positions", "features"]:
                with pytest.raises((TypeError, ValueError)):
                    self.ClassTested(**{**valid_kwargs, attr: torch.randn(num_pts + 5, 3)})


    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @with_seed(42)
    @pytest.mark.parametrize("device,other_device", [("cpu", "cuda"), ("cuda", "cpu")])
    @pytest.mark.parametrize("with_features", [False, True, ["a", "b"]])
    def test_to_device(self, device, other_device, with_features):
        instance = self.ClassTested(**self.sample_input_kwargs(10, with_features=with_features, device=device))
        assert instance.positions.device.type == device

        moved = instance.to(other_device)
        assert moved.positions.device.type == other_device

        cuda_instance = instance.cuda()
        assert cuda_instance.positions.device.type == "cuda"

        cpu_instance = cuda_instance.cpu()
        assert cpu_instance.positions.device.type == "cpu"

        half_instance = instance.to(dtype=torch.float16)
        for attr in half_instance.get_attributes(only_tensors=True):
            val = getattr(half_instance, attr)
            if isinstance(val, dict):
                for v in val.values():
                    assert v.dtype == torch.float16
            else:
                assert val.dtype == torch.float16


    @with_seed(42)
    def test_get_attributes(self):
        kwargs = self.sample_input_kwargs(10, with_features=True)
        instance = self.ClassTested(**kwargs)
        attrs = instance.get_attributes()
        for attr in attrs:
            assert getattr(instance, attr) is not None
        for key in kwargs:
            if key in self.ClassTested.class_tensor_attributes() + self.ClassTested.class_other_attributes():
                assert key in attrs, f"Expected {key} in get_attributes()"

    @with_seed(42)
    def test_describe_attribute(self):
        instance = self.ClassTested(**self.sample_input_kwargs(10, with_features=True))

        with pytest.raises(AttributeError):
            instance.describe_attribute("__nonexistent_attribute__")

        for attr in instance.get_attributes(only_tensors=True):
            result = instance.describe_attribute(attr)
            assert isinstance(result, str)
            long_result = instance.describe_attribute(attr, print_stats=True)
            assert isinstance(long_result, str)
            assert len(long_result) >= len(result)

    @with_seed(42)
    def test_to_string_and_print(self):
        kwargs = self.sample_input_kwargs(
            10, with_features=True, optional_attributes=self.class_optional_attributes)
        instance = self.ClassTested(**kwargs)
        s = instance.to_string()
        assert isinstance(s, str)
        for attr in instance.get_attributes(only_tensors=True):
            assert attr in s, f"Attribute {attr} missing from to_string output"
        assert str(instance) == s
        assert repr(instance) == s

        s_long = instance.to_string(print_stats=True)
        assert isinstance(s_long, str)
        assert len(s_long) > len(s)

        s_longer = instance.to_string(print_stats=True, detailed=True)
        assert isinstance(s_longer, str)
        assert len(s_longer) > len(s_long)

        instance_minimal = self.ClassTested(**self.sample_input_kwargs(10))
        s_minimal = instance_minimal.to_string()
        assert isinstance(s_minimal, str)

    def test_cat_valid(self, cat_input_kwargs):
        instances = [self.ClassTested(**kwargs) for kwargs in cat_input_kwargs]
        cat_instance = self.ClassTested.cat(instances)

        assert len(cat_instance) == sum(len(instance) for instance in instances), f"Length of cat instance does not match expected value"

        for attr in self.ClassTested.class_tensor_attributes():
            assert (cat_input_kwargs[0].get(attr) is None) == (getattr(cat_instance, attr) is None), f"Attribute {attr} has wrong set state in cat instance"

            if attr != "features" and cat_input_kwargs[0].get(attr) is not None:
                assert contained_torch_equal(
                    getattr(cat_instance, attr), torch.cat([kwargs[attr] for kwargs in cat_input_kwargs], dim=0), approximate=True), f"Attribute {attr} does not match expected value"

        if "features" in cat_input_kwargs[0]:
            assert cat_instance.features is not None, f"Features are not set in cat instance"

            if isinstance(cat_input_kwargs[0]["features"], dict):
                for feature, value in cat_input_kwargs[0]["features"].items():
                    assert contained_torch_equal(
                        cat_instance.features[feature], torch.cat([kwargs["features"][feature] for kwargs in cat_input_kwargs], dim=0), approximate=True), f"Feature {feature} does not match expected value"
            else:
                assert contained_torch_equal(
                    cat_instance.features, torch.cat([kwargs["features"] for kwargs in cat_input_kwargs], dim=0), approximate=True), f"Features do not match expected value"

        for attr in self.ClassTested.class_other_attributes():
            assert getattr(cat_instance, attr) == getattr(instances[0], attr), f"Attribute {attr} does not match expected value"

    def test_cat_permissive(self, cat_permissive_in_out):
        """
        Expects a fixture that is a tuple ([{constructor_kwargs0}, {constructor_kwargs1}..], {output_kwargs})
        """
        kwargs_list = cat_permissive_in_out[0]
        expected_out_kwargs = cat_permissive_in_out[1]

        inputs = [self.ClassTested(**kw) for kw in kwargs_list]
        with pytest.raises(ValueError):  # Fails if not permissive
            res = self.ClassTested.cat(inputs)
        res = self.ClassTested.cat(inputs, skip_errors=True)

        assert contained_torch_equal(res.as_dict(), expected_out_kwargs, approximate=True, print_error_context='')

    @with_seed(42)
    def test_cat_invalid(self):
        instance_with = self.ClassTested(**self.sample_input_kwargs(5, with_features=True))
        instance_without = self.ClassTested(**self.sample_input_kwargs(5, with_features=False))
        with pytest.raises(ValueError):
            self.ClassTested.cat([instance_with, instance_without])

    @with_seed(42)
    def test_cat_with_transforms(self, device, transform_matrix_info):
        """Transforms stored on models are applied (not concatenated) during cat."""
        transform_matrix, _, _ = transform_matrix_info
        transform_mat = transform_matrix.to(device)

        kwargs1 = self.sample_input_kwargs(5, device=device, sh_degree=1)
        kwargs2 = self.sample_input_kwargs(7, device=device, sh_degree=1)
        kwargs3 = self.sample_input_kwargs(4, device=device, sh_degree=1)

        inst1 = self.ClassTested(**kwargs1)
        inst1.transform = transform_mat
        inst2 = self.ClassTested(**kwargs2)
        inst2.transform = transform_mat.clone()
        inst3 = self.ClassTested(**kwargs3)  # no transform

        expected1 = inst1.as_transformed()
        expected2 = inst2.as_transformed()

        result = self.ClassTested.cat([inst1, inst2, inst3])

        assert result.transform is None, "cat result must have no stored transform"
        assert len(result) == 5 + 7 + 4, "total point count must match"

        expected_positions = torch.cat(
            [expected1.positions, expected2.positions, inst3.positions], dim=0)
        assert contained_torch_equal(
            result.positions, expected_positions, atol=1e-5, approximate=True), \
            "positions should reflect applied transforms"

        for attr in self.attributes_changed_by_transform():
            if attr == 'positions':
                continue
            v1 = getattr(expected1, attr)
            v2 = getattr(expected2, attr)
            v3 = getattr(inst3, attr)
            if torch.is_tensor(v1) and torch.is_tensor(v2) and torch.is_tensor(v3):
                expected_val = torch.cat([v1, v2, v3], dim=0)
                assert contained_torch_equal(
                    getattr(result, attr), expected_val, atol=1e-5, approximate=True), \
                    f"{attr} should reflect applied transforms"

    @with_seed(42)
    def test_getitem(self, device):
        for with_features in [False, True, ["a", "b"]]:
            kwargs = self.sample_input_kwargs(10, with_features=with_features, device=device)
            instance = self.ClassTested(**kwargs)
            mask = torch.zeros(10, dtype=torch.bool, device=device)
            mask[::2] = True

            sub = instance[mask]
            assert len(sub) == mask.sum().item()

            for attr in instance.get_attributes():
                val = getattr(sub, attr)
                orig_val = getattr(instance, attr)
                if attr in self.ClassTested.class_point_attributes():
                    if isinstance(orig_val, dict):
                        for k in orig_val:
                            assert contained_torch_equal(val[k], orig_val[k][mask], approximate=True)
                    else:
                        assert contained_torch_equal(val, orig_val[mask], approximate=True)
                elif attr in self.ClassTested.class_other_attributes():
                    assert val == orig_val

    @with_seed(42)
    def test_copy(self, device):
        kwargs = self.sample_input_kwargs(10, with_features=True, device=device)
        instance = self.ClassTested(**kwargs)
        copied = copy.copy(instance)

        for attr in instance.get_attributes():
            orig_val = getattr(instance, attr)
            copy_val = getattr(copied, attr)
            assert contained_torch_equal(orig_val, copy_val, approximate=True)
            if torch.is_tensor(orig_val):
                assert orig_val.data_ptr() == copy_val.data_ptr()
            elif isinstance(orig_val, dict):
                for k in orig_val:
                    if torch.is_tensor(orig_val[k]):
                        assert orig_val[k].data_ptr() == copy_val[k].data_ptr()

    @with_seed(42)
    def test_deepcopy(self, device):
        kwargs = self.sample_input_kwargs(10, with_features=True, device=device)
        instance = self.ClassTested(**kwargs)
        deep = copy.deepcopy(instance)

        for attr in instance.get_attributes():
            assert contained_torch_equal(
                getattr(instance, attr), getattr(deep, attr), approximate=True), f'Failed for {attr}'

        for attr in instance.get_attributes(only_tensors=True):
            orig_val = getattr(instance, attr)
            deep_val = getattr(deep, attr)
            if isinstance(deep_val, dict):
                for k in deep_val:
                    rv = torch.randn_like(deep_val[k])
                    deep_val[k][:] = rv
                    assert contained_torch_equal(getattr(deep, attr)[k], rv, approximate=True), \
                        f'Deep copy {attr}[{k}] should have the random value'
                    assert not contained_torch_equal(orig_val[k], rv, approximate=True), \
                        f'Original {attr}[{k}] should be unchanged'
            else:
                rv = torch.randn_like(deep_val)
                deep_val[:] = rv
                assert contained_torch_equal(getattr(deep, attr), rv, approximate=True), \
                    f'Deep copy {attr} should have the random value'
                assert not contained_torch_equal(orig_val, rv, approximate=True), \
                    f'Original {attr} should be unchanged'

    @with_seed(42)
    def test_sanity(self, device):
        num_pts = 10
        valid_kwargs = self.sample_input_kwargs(num_pts, device=device)
        instance = self.ClassTested(**valid_kwargs)
        assert instance.check_sanity()

        # -- positions attribute --
        bad = copy.copy(instance)
        bad.positions = None
        assert not bad.check_sanity(log_error=False), "positions=None should fail"

        bad = copy.copy(instance)
        bad.positions = "not a tensor"
        assert not bad.check_sanity(log_error=False), "positions=string should fail"

        bad = copy.copy(instance)
        bad.positions = torch.randn(num_pts, 2, device=device)
        assert not bad.check_sanity(log_error=False), "positions shape (N,2) should fail"

        # -- features attribute --
        bad = copy.copy(instance)
        bad.features = None
        assert bad.check_sanity(log_error=False), "features=None should pass"

        bad = copy.copy(instance)
        bad.features = "not valid"
        assert not bad.check_sanity(log_error=False), "features=string should fail"

        bad = copy.copy(instance)
        bad.features = 42
        assert not bad.check_sanity(log_error=False), "features=int should fail"

        bad = copy.copy(instance)
        bad.features = torch.randn(num_pts + 5, 3, device=device)
        assert not bad.check_sanity(log_error=False), "features tensor wrong num points should fail"

        bad = copy.copy(instance)
        bad.features = {"color": "not_a_tensor"}
        assert not bad.check_sanity(log_error=False), "features dict with non-tensor should fail"

        bad = copy.copy(instance)
        bad.features = {"color": torch.randn(num_pts + 5, 3, device=device)}
        assert not bad.check_sanity(log_error=False), "features dict wrong num points should fail"

    @with_seed(115)
    def test_detach(self):
        for with_features in [False, True, ["a", "b"]]:
            kwargs = self.sample_input_kwargs(10, with_features=with_features)
            instance = self.ClassTested(**kwargs)

            for attr in instance.get_attributes(only_tensors=True):
                val = getattr(instance, attr)
                if isinstance(val, dict):
                    for k in val:
                        val[k].requires_grad_(True)
                else:
                    val.requires_grad_(True)

            detached = instance.detach()

            for attr in detached.get_attributes():
                orig_val = getattr(instance, attr)
                det_val = getattr(detached, attr)
                if attr in self.ClassTested.class_point_attributes():
                    if isinstance(orig_val, dict):
                        for k in orig_val:
                            assert contained_torch_equal(det_val[k], orig_val[k], approximate=True)
                            assert not det_val[k].requires_grad
                    else:
                        assert contained_torch_equal(det_val, orig_val, approximate=True)
                        assert not det_val.requires_grad
                elif attr in self.ClassTested.class_tensor_attributes():
                    if isinstance(orig_val, dict):
                        for k in orig_val:
                            assert contained_torch_equal(det_val[k], orig_val[k], approximate=True)
                            assert not det_val[k].requires_grad
                    elif torch.is_tensor(det_val):
                        assert contained_torch_equal(det_val, orig_val, approximate=True)
                        assert not det_val.requires_grad
                elif attr in self.ClassTested.class_other_attributes():
                    assert det_val == orig_val


    @with_seed(42)
    @pytest.mark.parametrize("as_argument", [True, False])
    @pytest.mark.parametrize("batched", [True, False])
    def test_as_transformed(self, device, transform_matrix_info, as_argument, batched):
        transform_matrix, scale_orig, rot_orig = transform_matrix_info
        kwargs = self.sample_input_kwargs(10, with_features=True, device=device)
        instance = self.ClassTested(**kwargs)
        orig_instance = copy.deepcopy(instance)

        transform_mat = transform_matrix.to(device)
        inv_transform = torch.inverse(transform_mat)
        if batched:
            transform_mat = transform_mat.unsqueeze(0)
            inv_transform = inv_transform.unsqueeze(0)

        if as_argument:
            transformed = instance.as_transformed(transform_mat)
        else:
            instance.transform = transform_mat
            transformed = instance.as_transformed()

        assert set(transformed.get_attributes()) == set([x for x in orig_instance.get_attributes() if x != "transform"])
        assert transformed.transform is None, f'There should be no transform set on the transformed class'
        for attr in orig_instance.get_attributes():
            matches_orig = contained_torch_equal(getattr(transformed, attr), getattr(orig_instance, attr),
                                                 approximate=True)
            if attr in self.attributes_changed_by_transform():
                assert not matches_orig, f"Attribute {attr} was not transformed"
            else:
                assert matches_orig, f"Attribute {attr} changed after transformation (unexpected)"

            # TODO: check positions value

        assert transformed is not instance

        restored = transformed.as_transformed(inv_transform)

        orig_dict = orig_instance.as_dict()
        restored_dict = restored.as_dict()

        for k, orig_val in orig_dict.items():
            if k == 'orientations':  # Not ideal: test subclass should add this check
                # TODO: add compare utility for Gaussians
                diff = torch.min(torch.abs(orig_val - restored_dict[k]), torch.abs(orig_val + restored_dict[k]))
                assert diff.max() < 1e-4
            else:
                assert contained_torch_equal(orig_val, restored_dict[k], atol=1e-4, rtol=1e-4, approximate=True,
                                             print_error_context='')

    @with_seed(15)
    def test_as_transformed_both_or_none(self, device, transform_matrix_info):
        transform_matrix, scale_orig, rot_orig = transform_matrix_info
        kwargs = self.sample_input_kwargs(10, with_features=True, device=device)
        instance = self.ClassTested(**kwargs)

        transform_mat = transform_matrix.to(device)
        inv_transform = torch.inverse(transform_mat)

        # Case 1: neither self.transform nor additional_transform are set
        assert instance.transform is None
        result_none = instance.as_transformed()
        for attr in instance.get_attributes():
            assert contained_torch_equal(getattr(result_none, attr), getattr(instance, attr),
                                         approximate=True), f"Attribute {attr} changed with no transform"
        assert result_none is not instance

        # Case 2: both self.transform and additional_transform are set
        # Setting stored transform and passing argument should chain: additional @ stored
        instance.transform = transform_mat
        result_both = instance.as_transformed(inv_transform)
        assert result_both.transform is None

        orig_no_transform = self.ClassTested(**kwargs)
        for attr in orig_no_transform.get_attributes():
            orig_val = getattr(orig_no_transform, attr)
            restored_val = getattr(result_both, attr)
            if attr == 'orientations':  # Not ideal: test subclass should add this check
                # TODO: add compare utility for Gaussians
                diff = torch.min(torch.abs(orig_val - restored_val), torch.abs(orig_val + restored_val))
                assert diff.max() < 1e-4, f"Attribute {attr} not restored after chained inverse"
            else:
                assert contained_torch_equal(orig_val, restored_val, atol=1e-4, rtol=1e-4, approximate=True,
                                             print_error_context=''), \
                    f"Attribute {attr} not restored after chained inverse"
    

class TestPointSamples(BaseTests):
    ClassTested = PointSamples
    class_required_attributes = ["positions"]

    def sample_input_kwargs(self, num_pts, optional_attributes=None, with_features: bool | list[str] = False, device="cpu", dtype=torch.float32, **kwargs):
        kwargs = {
            "positions": torch.randn((num_pts, 3), device=device, dtype=dtype)
        }
        if with_features:
            kwargs["features"] = self.sample_features_input(num_pts, with_features, device=device, dtype=dtype)
        return kwargs

    def expected_tensor_attributes(self):
        return ["positions", "features", "transform"]

    def expected_point_attributes(self):
        return ["positions", "features"]

    def attributes_changed_by_transform(self):
        return ["positions"]

    @with_seed(15)
    @pytest.fixture(params=[False, True, ["magic", "weight", "size"]])
    def cat_input_kwargs(self, request, device):
        with_features = request.param
        
        kwargs_list = [
            self.sample_input_kwargs(5, device=device),
            self.sample_input_kwargs(8, device=device),
            self.sample_input_kwargs(3, device=device),
        ]

        feature_shape = self.sample_feature_shape()
        if with_features:
            kwargs_list[0]["features"] = self.sample_features_input(5, with_features, device=device, feature_shape=feature_shape)
            kwargs_list[1]["features"] = self.sample_features_input(8, with_features, device=device, feature_shape=feature_shape)
            kwargs_list[2]["features"] = self.sample_features_input(3, with_features, device=device, feature_shape=feature_shape)

        return kwargs_list

    @pytest.fixture(params=['total_miss', 'total_miss2', 'partial_miss', 'partial_miss2'])
    def cat_permissive_in_out(self, request, device):
        expected_kwargs = {}
        if request.param == 'total_miss':
            kwargs_list = [
                self.sample_input_kwargs(5, device=device),
                self.sample_input_kwargs(8, device=device, with_features=True)
            ]
        elif request.param == 'total_miss2':
            kwargs_list = [
                self.sample_input_kwargs(13, device=device, with_features=True),
                self.sample_input_kwargs(12, device=device, with_features=["f1", "f2"])
            ]
        elif request.param == 'partial_miss':
            kwargs_list = [
                self.sample_input_kwargs(13, device=device),
                self.sample_input_kwargs(8, device=device)
            ]
            kwargs_list[0]["features"] = self.sample_features_input(13, True, device=device,
                                                                        feature_shape=[3])
            kwargs_list[1]["features"] = self.sample_features_input(8, True, device=device,
                                                                    feature_shape=[4, 5])
        elif request.param == 'partial_miss2':
            kwargs_list = [
                self.sample_input_kwargs(5, device=device),
                self.sample_input_kwargs(7, device=device)
            ]
            kwargs_list[0]["features"] = { 'a': torch.randn((5, 15), device=device),
                                           'c': torch.randn((5, 3), device=device),
                                           'b': torch.randn((5, 3, 6), device=device)}
            kwargs_list[1]["features"] = { 'a': torch.randn((7, 12), device=device),
                                           'k':  torch.randn((7, 12), device=device),
                                           'b': torch.randn((7, 3, 6), device=device)}
            expected_kwargs['features'] = {'b': torch.cat([kw['features']['b'] for kw in kwargs_list], dim=0)}
        else:
            raise RuntimeError(f'Fixture param {request.param} not implemented properly')

        expected_kwargs['positions'] = torch.cat([kw['positions'] for kw in kwargs_list], dim=0)
        return kwargs_list, expected_kwargs

    
    def test_sanity_class_specific(self):
        # We don't need to implement it; base test already tests positions and features
        pass


class TestGaussians(BaseTests):
    ClassTested = GaussianSplatModel

    def expected_tensor_attributes(self):
        return ["positions", "features", "orientations", "scales", "opacities", "sh_coeff", "transform"]

    def expected_point_attributes(self):
        return ["positions", "features", "orientations", "scales", "opacities", "sh_coeff"]

    def attributes_changed_by_transform(self):
        return ["positions", "orientations", "scales", "sh_coeff"]

    def sample_input_kwargs(self, num_pts, optional_attributes=None, with_features: bool | list[str] = False, device="cpu", dtype=torch.float32, sh_degree=None):
        if sh_degree is None:
            sh_degree = random.randint(1, 4)

        kwargs = {}
        kwargs["positions"] = torch.randn((num_pts, 3), device=device, dtype=dtype)
        kwargs["orientations"] = torch.nn.functional.normalize(torch.randn((num_pts, 4), device=device, dtype=dtype), dim=-1)
        kwargs["scales"] = torch.randn((num_pts, 3), device=device, dtype=dtype)
        kwargs["opacities"] = torch.randn((num_pts,), device=device, dtype=dtype)
        kwargs["sh_coeff"] = torch.randn((num_pts, (sh_degree + 1) ** 2, 3), device=device, dtype=dtype)

        if optional_attributes is None:
            optional_attributes = []
        for attr in optional_attributes:
            if attr not in self.class_optional_attributes:
                raise ValueError(f"Add test support for optional attribute (new?): {attr}")

        if with_features:
            kwargs["features"] = self.sample_features_input(num_pts, with_features, device=device, dtype=dtype)
        return kwargs

    @with_seed(15)
    @pytest.fixture(params=[False, True, ["twist", "texture"]])
    def cat_input_kwargs(self, request, device):
        optional = [] if len(self.class_optional_attributes) == 0 else np.random.choice(self.class_optional_attributes, size=2, replace=False).tolist()
        
        with_features = request.param

        sh_degree = random.randint(1, 4)

        kwargs_list = [
            self.sample_input_kwargs(5, optional_attributes=optional, device=device, sh_degree=sh_degree),
            self.sample_input_kwargs(8, optional_attributes=optional, device=device, sh_degree=sh_degree),
            self.sample_input_kwargs(3, optional_attributes=optional, device=device, sh_degree=sh_degree)
        ]

        feature_shape = self.sample_feature_shape()
        if with_features:
            kwargs_list[0]["features"] = self.sample_features_input(5, with_features, device=device,
                                                                    feature_shape=feature_shape)
            kwargs_list[1]["features"] = self.sample_features_input(8, with_features, device=device,
                                                                    feature_shape=feature_shape)
            kwargs_list[2]["features"] = self.sample_features_input(3, with_features, device=device,
                                                                    feature_shape=feature_shape)
        return kwargs_list

    @pytest.fixture(params=['case1', 'case2'])
    def cat_permissive_in_out(self, request, device):
        expected_kwargs = {}
        if request.param == 'case1':
            kwargs_list = [
                self.sample_input_kwargs(5, sh_degree=2, device=device),
                self.sample_input_kwargs(10, sh_degree=3, device=device),
                self.sample_input_kwargs(8, sh_degree=4, device=device, with_features=True)
            ]
            expected_kwargs['sh_degree'] = 2
            min_shape = kwargs_list[0]['sh_coeff'].shape[1]
            expected_kwargs['sh_coeff'] = torch.cat(
                [kw['sh_coeff'][:, :min_shape, :] for kw in kwargs_list])
        elif request.param == 'case2':
            kwargs_list = [
                self.sample_input_kwargs(7, sh_degree=3, device=device),
                self.sample_input_kwargs(5, sh_degree=0, device=device)
            ]
            kwargs_list[1]["features"] = {'a': torch.randn((5, 15), device=device),
                                          'c': torch.randn((5, 3), device=device),
                                          'b': torch.randn((5, 3, 6), device=device)}
            kwargs_list[0]["features"] = {'a': torch.randn((7, 12), device=device),
                                          'k': torch.randn((7, 12), device=device),
                                          'b': torch.randn((7, 3, 6), device=device)}
            expected_kwargs['features'] = {'b': torch.cat([kw['features']['b'] for kw in kwargs_list], dim=0)}
            expected_kwargs['sh_degree'] = 0
            min_shape = kwargs_list[1]['sh_coeff'].shape[1]
            expected_kwargs['sh_coeff'] = torch.cat(
                [kw['sh_coeff'][:, :min_shape, :] for kw in kwargs_list])
        else:
            raise RuntimeError(f'Fixture param {request.param} not implemented properly')
        for k in self.ClassTested.class_tensor_attributes():
            if k not in ['features', 'sh_coeff'] and kwargs_list[0].get(k) is not None:
                expected_kwargs[k] = torch.cat([kw[k] for kw in kwargs_list], dim=0)
        return kwargs_list, expected_kwargs

    @with_seed(42)
    @pytest.mark.parametrize("as_argument", [True, False])
    def test_as_transformed_gaussians(self, device, transform_matrix_info, as_argument):
        transform_matrix, scale_orig, rot_orig = transform_matrix_info
        kwargs = self.sample_input_kwargs(10, with_features=True, device=device)
        instance = self.ClassTested(**kwargs)
        orig_instance = copy.deepcopy(instance)

        transform_mat = transform_matrix.to(device)

        if as_argument:
            transformed = instance.as_transformed(transform_mat)
        else:
            instance.transform = transform_mat
            transformed = instance.as_transformed()

        assert set(transformed.get_attributes()) == set([x for x in orig_instance.get_attributes() if x != "transform"])
        assert transformed.transform is None, f'There should be no transform set on the transformed class'
        # Check scale correctly applied (to check if log scale correctly not used)
        assert contained_torch_equal(transformed.scales, orig_instance.scales * scale_orig[0, 0], approximate=True)


    @pytest.mark.parametrize('attr', ['orientations', 'scales', 'opacities', 'sh_coeff'])
    @pytest.mark.parametrize('sh_degree', [0, 1, 3])
    def test_sanity_class_specific(self, sh_degree, caplog, attr):
        import logging
        num_pts = 10
        num_sh = (sh_degree + 1) ** 2
        valid_kwargs = self.sample_input_kwargs(num_pts, sh_degree=sh_degree)

        self.ClassTested(**valid_kwargs, strict_checks=True)
        self.ClassTested(**valid_kwargs, strict_checks=False)
        instance = self.ClassTested(**valid_kwargs)
        assert instance.check_sanity()

        n = len(instance)
        assert instance.positions.shape == (n, 3)
        assert instance.orientations.shape == (n, 4)
        assert instance.scales.shape == (n, 3)
        assert instance.opacities.shape == (n,)
        assert instance.sh_coeff.shape == (n, num_sh, 3)
        assert instance.sh_degree == sh_degree

        invalid_tensor_shapes = {
            'orientations': [
                torch.randn(num_pts, 3),
                torch.randn(num_pts + 5, 4),
            ],
            'scales': [
                torch.randn(num_pts, 2),
                torch.randn(num_pts + 5, 3),
            ],
            'opacities': [
                torch.randn(num_pts, 2),
                torch.randn(num_pts + 5),
            ],
            'sh_coeff': [
                torch.randn(num_pts, num_sh, 4),
                torch.randn(num_pts, num_sh + 10, 3),
                torch.randn(num_pts + 5, num_sh, 3),
            ],
            'transform': [
                torch.randn(3, 3),
                torch.randn(1, 2, 3),
            ]
        }

        if attr in ['sh_degree']:
            pass
        else:
            bad_values = invalid_tensor_shapes[attr]
            for bad_val in bad_values:
                bad_kwargs = {**valid_kwargs, attr: bad_val}
                if attr == 'sh_coeff':
                    bad_kwargs['sh_degree'] = num_sh  # needed for sanity check

                with pytest.raises(ValueError):
                    self.ClassTested(**bad_kwargs, strict_checks=True)

                bad_instance = self.ClassTested(**bad_kwargs, strict_checks=False)

                with caplog.at_level(logging.ERROR, logger='kaolin.rep.gaussians'):
                    caplog.clear()
                    assert not bad_instance.check_sanity(), \
                        f"{attr} shape {bad_val.shape} should fail check_sanity()"
                    assert any(attr in rec.message for rec in caplog.records), \
                        f"Log should mention '{attr}' for shape {bad_val.shape}"

                caplog.clear()
                assert not bad_instance.check_sanity(log_error=False), \
                    f"{attr} shape {bad_val.shape} should fail check_sanity(log_error=False)"
                assert len(caplog.records) == 0, \
                    f"No log messages expected with log_error=False for {attr}"

        if attr in ['sh_degree']:
            pass
        else:
            for bad_val in [None, "not a tensor"]:
                bad_instance = self.ClassTested(**valid_kwargs, strict_checks=False)
                setattr(bad_instance, attr, bad_val)

                with caplog.at_level(logging.ERROR, logger='kaolin.rep.gaussians'):
                    caplog.clear()
                    assert not bad_instance.check_sanity(), \
                        f"{attr}={bad_val!r} should fail check_sanity()"
                    assert any(attr in rec.message for rec in caplog.records), \
                        f"Log should mention '{attr}' for value {bad_val!r}"

                caplog.clear()
                assert not bad_instance.check_sanity(log_error=False), \
                    f"{attr}={bad_val!r} should fail check_sanity(log_error=False)"
                assert len(caplog.records) == 0, \
                    f"No log messages expected with log_error=False for {attr}={bad_val!r}"