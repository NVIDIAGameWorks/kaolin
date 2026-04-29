# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

import os
import pytest
import numpy as np
import random
import string
import torch

import kaolin.utils.testing

from kaolin.visualize.web.io import (
    BinaryIoDataType,
    ImageFormat,
    MESSAGE_CONTENT_KEY,
    MESSAGE_TAG_KEY,
    gap_until_offset_n,
    gap_until_offset_4,
    string_from_binary,
    string_to_binary,
    np_type_from_type_id,
    value_to_type,
    convert_value_to_supported_format,
    value_from_binary,
    value_to_binary,
    named_value_from_binary,
    named_value_to_binary,
    encode_message,
    to_binary,
    from_binary
)

_NON_ARRAY_TYPES = (BinaryIoDataType.STRING, BinaryIoDataType.DICT, BinaryIoDataType.LIST,
                    BinaryIoDataType.PNG, BinaryIoDataType.JPEG)
_types_used_for_testing = [x for x in list(BinaryIoDataType)
                           if x not in (BinaryIoDataType.UNSUPPORTED,
                                        BinaryIoDataType.PNG, BinaryIoDataType.JPEG)]
_array_types_used_for_testing = [x for x in _types_used_for_testing if x not in _NON_ARRAY_TYPES]


def generate_random_string(length=None, is_ascii=False):
    """
    Generate a random string for testing.

    Args:
        length: String length (random if None)
        is_ascii: If True, generate ASCII-only string; if False, always include UTF-8 characters

    Returns:
        str: Random string
    """
    if length is None:
        length = random.randint(4, 15)

    if length == 0:
        return ''

    ascii_chars = [
            *string.ascii_letters,
            *string.digits,
            '.', ',', '!', '?', ':', ';', '-', '_', '(', ')', '[', ']']
    utf8_chars = [
            # Accented characters
            'á', 'é', 'í', 'ó', 'ú', 'ñ', 'ü', 'ç',
            'à', 'è', 'ì', 'ò', 'ù', 'â', 'ê', 'î', 'ô', 'û',
            'ä', 'ë', 'ï', 'ö', 'ÿ', 'å', 'æ', 'ø',
            # Greek letters
            'α', 'β', 'γ', 'δ', 'ε', 'λ', 'μ', 'π', 'σ', 'φ', 'ω',
            # Mathematical symbols
            '±', '×', '÷', '≤', '≥', '≠', '∞', '∑', '∫',
            # Currency symbols
            '€', '£', '¥', '¢',
            # Emojis (basic set)
            '😀', '😎', '🚀', '🎉', '❤️', '🌟', '🔥', '💡'
        ]

    if is_ascii:
       selected_chars = random.choices(ascii_chars, k=length)
    else:
        # Ensure utf8 characters are present if is_ascii is False
        num_utf8 = random.randint(1, length - 2)
        selected_chars = random.choices(ascii_chars, k=length-num_utf8)
        selected_chars += random.choices(utf8_chars, k=num_utf8)

    random.shuffle(selected_chars)
    return ''.join(selected_chars)


def generate_random_np_array(value_type: BinaryIoDataType):
    np_type, _ = np_type_from_type_id(value_type)
    try:
        np_type_info = np.iinfo(np_type)
    except ValueError as e:
        np_type_info = np.finfo(np_type)

    shape_length = random.randint(1, 4)
    shape = [random.randint(1, 10) for x in range(shape_length)]

    if np_type in [np.int8, np.uint8, np.int16, np.int32, np.uint32, np.int64]:
        value = np.random.default_rng().integers(np_type_info.min, np_type_info.max, size=shape, dtype=np_type)
    elif np_type in [np.float16]:
        value = np.random.default_rng().random(shape, dtype=np.float32).astype(np_type)
    else:
        value = np.random.default_rng().standard_normal(shape, dtype=np_type)

    return value


def generate_random_torch_array(value_type: BinaryIoDataType):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    value = generate_random_np_array(value_type)
    return torch.from_numpy(value).to(device)


def generate_random_value(value_type: BinaryIoDataType, included_types=_types_used_for_testing,
                          ascii_only=False):
    if value_type == BinaryIoDataType.STRING:
        value = generate_random_string(is_ascii=ascii_only)
    elif value_type == BinaryIoDataType.DICT:
        # Avoid infinite recursion
        non_dict_types = [t for t in included_types if t != BinaryIoDataType.DICT]
        value = generate_random_dict(included_types=non_dict_types,
                                    ascii_only=ascii_only)
    elif value_type == BinaryIoDataType.LIST:
        num_entries = random.randint(1, 10)
        non_list_types = [t for t in included_types if t != BinaryIoDataType.LIST]
        value = [generate_random_value(value_type = random.choice(non_list_types)) for x in range(num_entries)]
    else:
        value = generate_random_torch_array(value_type)
    return value
    


def generate_random_dict(num_entries=None,
                         included_types=[x for x in _types_used_for_testing if x != BinaryIoDataType.DICT],
                         ascii_only=False):
    """
    Generate a random dictionary with various data types for testing.

    Args:
        num_entries: Number of entries (random if None)

    Returns:
        dict: Dictionary with random keys and values of various types
    """
    if num_entries is None:
        num_entries = random.randint(1, 10)

    result = {}

    selected_types = []
    if num_entries >= len(included_types):
        selected_types = included_types
    if len(selected_types) < num_entries:
        selected_types += list(np.random.choice(included_types, num_entries - len(selected_types)))
    assert len(selected_types) == num_entries

    for idx, value_type in enumerate(selected_types):
        # Generate random key
        # Always include one ascii key
        key = None
        while key is None or key in result:
            if not ascii_only and idx == 0 and num_entries > 1:
                key = generate_random_string(is_ascii=True)
            else:
                key = generate_random_string(is_ascii=ascii_only)

        # Generate random value based on random type
        value_type = random.choice(included_types)
        value = generate_random_value(value_type, included_types=included_types, ascii_only=ascii_only)
        result[key] = value
    return result


class TestAlignment:
    """Test alignment utility functions."""

    def test_gap_util_offset_n(self):
        """Test gap calculation for N-byte alignment."""
        for bytesize in [2, 4, 8, 16]:
            for expected_gap in range(0, bytesize):
                n = random.randint(0, 5)
                offset = bytesize * n - expected_gap
                actual_gap = gap_until_offset_n(offset, bytesize)
                assert expected_gap == actual_gap, f"gap_util_offset_n({offset}, {bytesize}) = {actual_gap}, expected {expected_gap}"

    def test_gap_util_offset_4(self):
        """Test gap calculation for 4-byte alignment."""
        bytesize = 4
        for expected_gap in range(0, bytesize):
            n = random.randint(0, 5)
            offset = bytesize * n - expected_gap
            actual_gap = gap_until_offset_4(offset)
            assert expected_gap == actual_gap, f"gap_util_offset_4({offset}) = {actual_gap}, expected {expected_gap}"


class TestTypeMapping:
    """Test type mapping and conversion functions."""

    @pytest.mark.parametrize('type_id', _array_types_used_for_testing)
    def test_np_type_from_type_id(self, type_id: BinaryIoDataType):
        """Test NumPy type lookup from binary IO type ID."""
        actual_np_type, actual_bytes_per_element = np_type_from_type_id(type_id)
        if type_id == BinaryIoDataType.INT8:
            expected_np_type = np.int8
            expected_bytes_per_element = 1
        elif type_id == BinaryIoDataType.UINT8:
            expected_np_type = np.uint8
            expected_bytes_per_element = 1
        elif type_id == BinaryIoDataType.INT16:
            expected_np_type = np.int16
            expected_bytes_per_element = 2
        elif type_id == BinaryIoDataType.INT32:
            expected_np_type = np.int32
            expected_bytes_per_element = 4
        elif type_id == BinaryIoDataType.UINT32:
            expected_np_type = np.uint32
            expected_bytes_per_element = 4
        elif type_id == BinaryIoDataType.INT64:
            expected_np_type = np.int64
            expected_bytes_per_element = 8
        elif type_id == BinaryIoDataType.FLOAT16:
            expected_np_type = np.float16
            expected_bytes_per_element = 2
        elif type_id == BinaryIoDataType.FLOAT32:
            expected_np_type = np.float32
            expected_bytes_per_element = 4
        elif type_id == BinaryIoDataType.FLOAT64:
            expected_np_type = np.float64
            expected_bytes_per_element = 8
        else:
            raise AssertionError(f'Test does not check expected types for {type_id}')
        assert actual_np_type == expected_np_type, f"np_type_from_type_id({type_id}) = {actual_np_type}, expected {expected_np_type}"
        assert actual_bytes_per_element == expected_bytes_per_element, f"np_type_from_type_id({type_id}) = {actual_bytes_per_element}, expected {expected_bytes_per_element}"

    def test_value_to_type(self):
        """Test type detection from values."""
        # Default-style probing: image detection is gated off, so type
        # detection is purely dtype-driven.
        raw = ImageFormat.RAW
        expected = BinaryIoDataType.INT8
        assert value_to_type(np.zeros((3, 5), dtype=np.int8), image_format=raw) == expected

        expected = BinaryIoDataType.UINT8
        assert value_to_type(np.zeros((6, 1, 1), dtype=np.uint8), image_format=raw) == expected

        expected = BinaryIoDataType.INT16
        assert value_to_type(np.zeros((14, 1), dtype=np.int16), image_format=raw) == expected

        expected = BinaryIoDataType.INT32
        assert value_to_type(np.zeros((3, 10), dtype=np.int32), image_format=raw) == expected
        assert value_to_type(convert_value_to_supported_format([1, 2, 3]), image_format=raw) == expected

        expected = BinaryIoDataType.UINT32
        assert value_to_type(np.zeros((3, 10), dtype=np.uint32), image_format=raw) == expected

        expected = BinaryIoDataType.INT64
        assert value_to_type(np.zeros((3, 10), dtype=np.int64), image_format=raw) == expected

        expected = BinaryIoDataType.FLOAT16
        assert value_to_type(np.zeros((3, 10), dtype=np.float16), image_format=raw) == expected

        expected = BinaryIoDataType.FLOAT32
        assert value_to_type(np.zeros((3, 10), dtype=np.float32), image_format=raw) == expected
        assert value_to_type(convert_value_to_supported_format([0.5, 3, 1]), image_format=raw) == expected

        expected = BinaryIoDataType.FLOAT64
        assert value_to_type(np.zeros((3, 10), dtype=np.float64), image_format=raw) == expected

        expected = BinaryIoDataType.STRING
        assert value_to_type('hello world', image_format=raw) == expected

        expected = BinaryIoDataType.DICT
        assert value_to_type({'a': 5, 'b': torch.zeros((3,))}, image_format=raw) == expected

        expected = BinaryIoDataType.LIST  # mixed or non-number lists are LIST
        assert value_to_type([1, 'a', {'c': 15}], image_format=raw) == expected

        expected = BinaryIoDataType.UNSUPPORTED
        for value in [None, 3.5, 1, {3, 4}]:
            assert value_to_type(value, image_format=raw) == expected

    def test_value_to_type_with_image_format(self):
        """Image-encodable detection must lock to the codec's channel set.

        Regression guard for a previous bug where the JPEG branch in
        :func:`value_to_type` was accidentally querying with the PNG
        channel set, making 4-channel uint8 inputs report as JPEG.
        That JPEG payload would then fail (or, before the alpha-drop
        fixup, silently lose alpha). After the fix, a 4-channel uint8
        image with ``image_format='jpeg'`` must fall through to UINT8.
        """
        # uint8 HWC with 3 channels: encodable as both PNG and JPEG.
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        assert value_to_type(rgb, image_format=ImageFormat.RAW) == BinaryIoDataType.UINT8
        assert value_to_type(rgb, image_format=ImageFormat.PNG) == BinaryIoDataType.PNG
        assert value_to_type(rgb, image_format=ImageFormat.JPEG) == BinaryIoDataType.JPEG

        # uint8 HWC with 4 channels: PNG-only. JPEG MUST fall through
        # to UINT8 since JPEG cannot express alpha.
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        assert value_to_type(rgba, image_format=ImageFormat.RAW) == BinaryIoDataType.UINT8
        assert value_to_type(rgba, image_format=ImageFormat.PNG) == BinaryIoDataType.PNG
        assert value_to_type(rgba, image_format=ImageFormat.JPEG) == BinaryIoDataType.UINT8, (
            "Bug guard: 4-channel uint8 with image_format='jpeg' must NOT be "
            "reported as JPEG (JPEG accepts only 1 or 3 channels)."
        )

        # Float images are never image-encodable; behavior must match RAW.
        float_img = np.zeros((8, 8, 3), dtype=np.float32)
        assert value_to_type(float_img, image_format=ImageFormat.PNG) == BinaryIoDataType.FLOAT32
        assert value_to_type(float_img, image_format=ImageFormat.JPEG) == BinaryIoDataType.FLOAT32

        # Non-image-shaped uint8 ndarray must not be detected as image.
        non_img = np.zeros((5,), dtype=np.uint8)
        assert value_to_type(non_img, image_format=ImageFormat.PNG) == BinaryIoDataType.UINT8
        assert value_to_type(non_img, image_format=ImageFormat.JPEG) == BinaryIoDataType.UINT8

    def test_convert_value_to_supported_format(self):
        """Test value conversion to supported formats."""
        # TODO: Test conversion of ints, floats, torch tensors, numpy arrays
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        def _check_convert_numeric_type(expected_io_type: BinaryIoDataType,
                                        accepted_torch_value: torch.Tensor,
                                        expected_np_type: np.dtype,
                                        accepted_num_value=None):
            # Test conversion of:
            # torch GPU tensor
            # torch cpu tensor
            # numpy tensor
            # primitive numeric type (if provided)
            #  --> all to same expected_np_type
            #  --> such that its detected expected_io_type is as expected
            accepted_values = [accepted_torch_value, accepted_torch_value.cpu(),
                               accepted_torch_value.cpu().numpy().astype(expected_np_type)] + \
                               ([accepted_num_value] if accepted_num_value is not None else [])
            for value in accepted_values:
                converted_value = convert_value_to_supported_format(value)
                failure_string = f"Converted {kaolin.utils.testing.tensor_info(value)} to unexpected value {kaolin.utils.testing.tensor_info(converted_value)}"
                assert isinstance(converted_value, np.ndarray), failure_string
                assert converted_value.dtype == expected_np_type, failure_string + f" - got unexpected dtype {converted_value.dtype} instead of {expected_np_type}"
                io_type = value_to_type(converted_value, image_format=ImageFormat.RAW)
                assert io_type == expected_io_type, failure_string + f" - detected unexpectedtype {io_type.name} instead of {expected_io_type.name}"

        _check_convert_numeric_type(
            BinaryIoDataType.INT8,
            torch.zeros((4, 7), dtype=torch.int8, device=device),
            np.int8)

        _check_convert_numeric_type(
            BinaryIoDataType.UINT8,
            torch.zeros((3, 5), dtype=torch.uint8, device=device),
            np.uint8)

        _check_convert_numeric_type(
            BinaryIoDataType.INT16,
            torch.zeros((2, 8), dtype=torch.int16, device=device),
            np.int16)

        _check_convert_numeric_type(
            BinaryIoDataType.INT32,
            torch.zeros((6, 3), dtype=torch.int32, device=device),
            np.int32,
            42)  # Test with primitive int
        
        _check_convert_numeric_type(
            BinaryIoDataType.INT32,
            torch.zeros((6, 3), dtype=torch.int32, device=device),
            np.int32,
            [42, 5, 7])  # Test with int list
        
        _check_convert_numeric_type(
            BinaryIoDataType.UINT32,
            torch.zeros((6, 3), dtype=torch.uint32, device=device),
            np.uint32)

        _check_convert_numeric_type(
            BinaryIoDataType.INT64,
            torch.zeros((1, 9), dtype=torch.int64, device=device),
            np.int64)

        _check_convert_numeric_type(
            BinaryIoDataType.FLOAT16,
            torch.zeros((5, 2), dtype=torch.float16, device=device),
            np.float16)

        _check_convert_numeric_type(
            BinaryIoDataType.FLOAT32,
            torch.zeros((3, 4), dtype=torch.float32, device=device),
            np.float32,
            3.14)  # Test with primitive float
        
        _check_convert_numeric_type(
            BinaryIoDataType.FLOAT32,
            torch.zeros((3, 4), dtype=torch.float32, device=device),
            np.float32,
            [3.14, 0, 1.5])  # Test with float list

        _check_convert_numeric_type(
            BinaryIoDataType.FLOAT64,
            torch.zeros((2, 6), dtype=torch.float64, device=device),
            np.float64)

        # Test string conversion
        test_strings = ["hello", "世界", "🚀emoji test", "", "mixed ASCII + UTF-8: café"]
        for test_string in test_strings:
            converted_value = convert_value_to_supported_format(test_string)
            assert isinstance(converted_value, str), f"String conversion failed for: '{test_string}'"
            assert converted_value == test_string, f"String changed during conversion: '{test_string}' -> '{converted_value}'"
            io_type = value_to_type(converted_value, image_format=ImageFormat.RAW)
            assert io_type == BinaryIoDataType.STRING, f"String type detection failed for: '{test_string}'"

        value = generate_random_dict()
        converted_value = convert_value_to_supported_format(value)
        assert isinstance(converted_value, dict)

        value = [1, 'a', {'a': 15}]
        converted_value = convert_value_to_supported_format(value)
        assert isinstance(converted_value, list)


class TestStringConversion:
    """Test string encoding/decoding functions."""

    @pytest.mark.parametrize("is_ascii", [True, False])
    def test_string_to_from_binary(self, is_ascii):
        """Test string to binary encoding and decoding roundtrip."""
        for offset in [0, 4, 15]:
            for length in [0, 5, 20, 64, 113]:
                string = generate_random_string(length=length, is_ascii=is_ascii)
                binary = string_to_binary(string)
                string_bytes = len(binary)
                if offset > 0:
                    binary = bytes(offset) + binary
                decoded = string_from_binary(binary, offset, string_bytes)
                assert string == decoded, f"Failed roundtrip for string: '{string}' (ascii={is_ascii}, offset={offset}, length={length})"


class TestBinaryValueIO:
    """Test binary value encoding/decoding functions for dictionary values that
    contain extra metadata, such as type id, shape length and shape."""

    @pytest.mark.parametrize("offset", [0, 3, 4, 16, 256])
    def test_value_to_from_binary_string(self, offset):
        """Test string encoding/decoding with extra metadata."""
        test_strings = [generate_random_string(10, is_ascii=True),
                        generate_random_string(15, is_ascii=False)]

        for test_string in test_strings:
            binary = value_to_binary(test_string, initial_offset=offset, image_format=ImageFormat.RAW)
            string_bytes = len(binary)
            if offset > 0:
                binary = bytes(offset) + binary
                binary += bytes(offset)  # also add at the end

            decoded, read_bytes = value_from_binary(binary, offset)
            assert test_string == decoded, f'Expected {test_string}, got {decoded}'
            assert string_bytes == read_bytes, f'Expected {string_bytes} bytes to be read, got {read_bytes} for string value {test_string}'


    @pytest.mark.parametrize("offset", [0, 1, 4, 8, 128])
    @pytest.mark.parametrize("value_type", _array_types_used_for_testing)
    def test_value_to_from_binary_array(self, value_type: BinaryIoDataType, offset):
        value = generate_random_torch_array(value_type)
        info_str = kaolin.utils.testing.tensor_info(value, 'torch tensor')
        same_value2 = value.cpu()
        info_str2 = kaolin.utils.testing.tensor_info(value, 'torch cpu tensor')
        same_value3 = value.cpu().numpy()
        info_str3 = kaolin.utils.testing.tensor_info(value, 'numpy')
        binary = value_to_binary(value, initial_offset=offset, image_format=ImageFormat.RAW)
        binary2 = value_to_binary(same_value2, initial_offset=offset, image_format=ImageFormat.RAW)
        binary3 = value_to_binary(same_value3, initial_offset=offset, image_format=ImageFormat.RAW)
        assert binary2 == binary, f'equivalent data encoded to different binaries: {info_str2} AND {info_str}'
        assert binary3 == binary, f'equivalent data encoded to different binaries: {info_str3} AND {info_str}'
        encoded_bytes = len(binary)
        if offset > 0:
            binary = bytes(offset) + binary
            binary += bytes(offset)  # Also add at the end
        decoded, read_bytes = value_from_binary(binary, offset)
        assert encoded_bytes == read_bytes, f'Expected {encoded_bytes} bytes to be read, got {read_bytes} for input {info_str}'
        kaolin.utils.testing.check_tensor(decoded, shape=value.shape, dtype=value.dtype, throw=True)
        kaolin.utils.testing.check_allclose(same_value2, decoded)


class TestNamedValueIO:
    """Test named value encoding/decoding functions."""

    def assert_roundtrip(self, name, value, offset):
        info_str = kaolin.utils.testing.tensor_info(value, f'input {name}')

        binary = named_value_to_binary(name, value, initial_offset=offset, image_format=ImageFormat.RAW)
        encoded_bytes = len(binary)
        if offset > 0:
            binary = bytes(offset) + binary + bytes(offset)
        decoded_name, decoded_value, read_bytes = named_value_from_binary(binary, offset)
        assert encoded_bytes == read_bytes, f'Expected {encoded_bytes} bytes to be read, got {read_bytes} for input {info_str}'
        assert isinstance(decoded_name, str)
        assert decoded_name == name

        assert kaolin.utils.testing.contained_torch_equal(
                value, decoded_value, approximate=True, ignore_device=True, print_error_context='')


    @pytest.mark.parametrize("offset", [0, 3, 8, 128])  # has to be 4-divisible
    def test_named_value_to_from_binary_dict(self, offset):
        value = {'cats 🤩': 'Сима и Таня'}
        name = 'test_dict'
        self.assert_roundtrip(name, value, offset)

        value = {'cats 🤩': 'Сима и Таня',
                 'photo': {'Sima': torch.rand((25,25), dtype=torch.float32),
                           'Tanya': (torch.rand((20, 40)).clip(0, 1) * 255).to(torch.uint8)}}
        name = '🌟kitties🌟'
        self.assert_roundtrip(name, value, offset)

    @pytest.mark.parametrize("offset", [0, 3, 8, 128])  # has to be 4-divisible
    @pytest.mark.parametrize("value_type", _types_used_for_testing)
    def test_named_value_to_from_binary(self, value_type, offset):
        """Test named value encoding to binary."""
        name = generate_random_string()
        assert len(name) > 0
        if value_type == BinaryIoDataType.STRING:
            value = generate_random_string(is_ascii=False)
            assert len(value) > 0
        elif value_type == BinaryIoDataType.DICT:
            value = generate_random_dict(5)
        else:
            value = generate_random_torch_array(value_type)

        self.assert_roundtrip(name, value, offset)

class TestDictIO:
    """Test dictionary encoding/decoding functions."""

    def assert_dicts_equal(self, in_dict, decoded):
        assert len(in_dict) > 0
        assert len(decoded) == len(in_dict)
        assert kaolin.utils.testing.contained_torch_equal(
            in_dict, decoded, approximate=True, ignore_device=True, print_error_context='')

        # in case something is wrong with contained_torch_equal
        encoded = to_binary(in_dict)
        encoded2 = to_binary(decoded)
        assert encoded == encoded2

    @pytest.fixture
    def dict_compatible(self):
        return generate_random_dict(12)

    @pytest.fixture
    def dict_np(self):
        res = generate_random_dict(5)
        res['np_arr1'] = generate_random_np_array(BinaryIoDataType.UINT8)
        res['np_arr2'] = generate_random_np_array(BinaryIoDataType.FLOAT32)
        return res

    @pytest.fixture
    def dict_primitive(self):
        res = generate_random_dict(3)
        res['num_int'] = 15
        res['num_float'] = 3.14
        res['list_mixed'] = [1, 'a']
        res['list'] = [1.5, 2, 3]
        res['list_empty'] = []
        return res

    @pytest.mark.parametrize("image_format", ['raw', 'png', 'jpeg'])
    def test_basic(self, image_format):
        dict0 = {'sima': 'cat', 'tanya': 'kitty'}
        baseline = to_binary(dict0)
        encoded = to_binary(dict0, image_format=image_format)
        # No image-shaped values: bytes match the raw default exactly.
        assert encoded == baseline, \
            f"image_format='{image_format}' altered bytes for non-image dict"
        decoded = from_binary(encoded)
        assert dict0.keys() == decoded.keys()
        assert dict0['sima'] == decoded['sima']
        assert dict0['tanya'] == decoded['tanya']

        # 1x1x2 float tensor is not image-shaped (last dim 2 not in {1,3,4}).
        dict1 = {'test': torch.tensor([16.5, 13.5], dtype=torch.float32).reshape(1, 1, 2),
                 'ñame': 'Andromedä'}
        baseline = to_binary(dict1)
        encoded = to_binary(dict1, image_format=image_format)
        assert encoded == baseline, \
            f"image_format='{image_format}' altered bytes for non-image dict"
        decoded = from_binary(encoded)
        assert dict1.keys() == decoded.keys()
        kaolin.utils.testing.check_tensor(decoded['test'], shape=dict1['test'].shape,
                                          dtype=dict1['test'].dtype, throw=True)
        assert torch.allclose(dict1['test'], decoded['test'])
        assert dict1['ñame'] == decoded['ñame']

    def test_to_from_binary_compatible(self, dict_compatible):
        """Test dictionary encoding to binary."""
        print(kaolin.utils.testing.tensor_info(dict_compatible))
        in_dict = dict_compatible
        encoded = to_binary(in_dict)
        decoded = from_binary(encoded)
        self.assert_dicts_equal(in_dict, decoded)

    def test_to_from_binary_np(self, dict_np):
        in_dict = dict_np
        encoded = to_binary(in_dict)
        decoded = from_binary(encoded)

        assert set(in_dict.keys()) == set(decoded.keys())
        keys_special = ['np_arr1', 'np_arr2']
        keys_other = [k for k in in_dict.keys() if k not in keys_special]
        # Standard comparison
        self.assert_dicts_equal(
            {k: in_dict[k] for k in keys_other},
            {k: decoded[k] for k in keys_other})
        # Input np arrays should now be torch
        self.assert_dicts_equal(
            {k: torch.from_numpy(in_dict[k]) for k in keys_special},
            {k: decoded[k] for k in keys_special})

    def test_to_from_binary_primitive(self, dict_primitive):
        in_dict = dict_primitive
        encoded = to_binary(in_dict)
        decoded = from_binary(encoded)

        assert set(in_dict.keys()) == set(decoded.keys())
        keys_special = ['list']
        keys_other = [k for k in in_dict.keys() if k not in keys_special]
        # Standard comparison
        self.assert_dicts_equal(
            {k: in_dict[k] for k in keys_other},
            {k: decoded[k] for k in keys_other})
        kaolin.utils.testing.check_tensor(decoded['list'], shape=(3,))
        assert torch.allclose(decoded['list'].float(), torch.tensor(in_dict['list'], dtype=torch.float32))

    def test_dict_roundtrip_multiple_random(self):
        """Test roundtrip with multiple random dictionaries.

        Pinned to ``image_format='raw'`` for lossless, shape-preserving
        deep equality: random uint8 arrays may be image-shaped and would
        otherwise be re-encoded (and reshaped, e.g. ``(H, W)`` -> ``(H, W, 1)``)
        under the default image format."""
        for _ in range(10):
            test_dict = generate_random_dict()
            encoded = to_binary(test_dict, image_format='raw')
            decoded = from_binary(encoded)
            self.assert_dicts_equal(test_dict, decoded)

    def test_dict_unsupported_exception(self):
        """Test that encoding dictionaries with unsupported types throws ValueError."""
        test_dict = generate_random_dict()
        test_dict['unsupported'] = {'a', 'b', 'c'}  # Set is unsupported type
        
        with pytest.raises(ValueError): #, match=r"Cannot encode value of type.*set.*to binary"):
            to_binary(test_dict)

    def test_nested_dicts(self):
        camera = kaolin.render.easy_render.default_camera()
        in_dict = camera.as_dict()
        encoded = to_binary(in_dict)
        decoded = from_binary(encoded)
        out_camera = kaolin.render.camera.Camera.from_dict(decoded)
        self.assert_dicts_equal(in_dict, out_camera.as_dict())


class TestListIO:
    """Test list encoding/decoding functions."""

    def assert_lists_equal(self, in_list, decoded):
        assert len(in_list) > 0
        assert len(decoded) == len(in_list)
        assert kaolin.utils.testing.contained_torch_equal(
            in_list, decoded, approximate=True, ignore_device=True, print_error_context='')

        # in case something is wrong with contained_torch_equal
        encoded = to_binary(in_list)
        encoded2 = to_binary(decoded)
        assert encoded == encoded2

    def test_primitive_int(self):
        """Test list with integers."""
        in_list = [1, 2, 3, 4, 5]
        encoded = to_binary(in_list)
        decoded = from_binary(encoded)
        
        # Homogeneous int list should be converted to torch tensor
        kaolin.utils.testing.check_tensor(decoded, shape=(5,), dtype=torch.int32, throw=True)
        assert torch.allclose(decoded.float(), torch.tensor(in_list, dtype=torch.float32))

    def test_primitive_float(self):
        """Test list with floats."""
        in_list = [1.5, 2.5, 3.14, 4.2, 5.7]
        encoded = to_binary(in_list)
        decoded = from_binary(encoded)
        
        # Homogeneous float list should be converted to torch tensor
        kaolin.utils.testing.check_tensor(decoded, shape=(5,), dtype=torch.float32, throw=True)
        assert torch.allclose(decoded, torch.tensor(in_list, dtype=torch.float32))

    def test_mixed(self):
        """Test list with mixed types."""
        in_list = [42, "hello world", 3.14, [1, 2, 3]]
        encoded = to_binary(in_list)
        decoded = from_binary(encoded)
        
        # Mixed-type list should remain as list
        assert isinstance(decoded, list)
        assert len(decoded) == len(in_list)
        
        # Check each element
        assert decoded[0] == in_list[0]  # int
        assert decoded[1] == in_list[1]  # string
        assert abs(decoded[2] - in_list[2]) < 0.001  # float (approximate)
        kaolin.utils.testing.check_allclose(decoded[3], torch.tensor(in_list[3], dtype=torch.int32))  # tensor

    def test_random_roundtrip(self):
        """Strict roundtrip: random lists encoded with ``image_format='raw'``
        must round-trip back to ``contained_torch_equal``-equal values
        (deep, lossless equality), since RAW disables image detection."""
        for _ in range(10):
            test_list = generate_random_value(BinaryIoDataType.LIST)
            encoded = to_binary(test_list, image_format='raw')
            decoded = from_binary(encoded)
            self.assert_lists_equal(test_list, decoded)

    @pytest.mark.parametrize("image_format", ['raw', 'png', 'jpeg'])
    def test_random_roundtrip_with_image_format(self, image_format):
        """Roundtrip random lists for every supported ``image_format``.

        With ``'raw'`` the output is deterministic and byte-identical to an
        explicit raw encode. With ``'png'``/``'jpeg'`` random uint8 arrays of
        image-like shapes may be re-encoded as compressed images (which is
        fine — this test only checks that the kwarg never breaks the
        round-trip)."""
        for _ in range(10):
            test_list = generate_random_value(BinaryIoDataType.LIST)
            baseline = to_binary(test_list, image_format='raw')
            encoded = to_binary(test_list, image_format=image_format)
            if image_format == 'raw':
                assert encoded == baseline, \
                    f"image_format='raw' must produce byte-identical bytes"
            decoded = from_binary(encoded)
            # JPEG is lossy and may reshape via image detection; a
            # successful round-trip (no exceptions, valid decode) is
            # sufficient for non-RAW formats.
            assert decoded is not None


# ---------------------------------------------------------------------------
# Image io tests (PNG / JPEG)
# ---------------------------------------------------------------------------

# Note: torchvision (and Pillow for the RGBA-PNG fallback) are required for
# the image io path. The repo as a whole already requires torchvision, so we
# do not pytest.importorskip it here — a missing torchvision is a test-env
# bug, not a feature gate.


_GOLDEN_FIXTURE_DIR = os.path.join('tests', 'samples', 'visualize')
_GOLDEN_FIXTURE_BIN = 'checkerboard.png.bin'
_GOLDEN_FIXTURE_PNG = 'checkerboard.png'


def _golden_fixture_dir():
    """Absolute path to the cross-language fixture directory."""
    repo_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..', '..', '..'))
    return os.path.join(repo_root, _GOLDEN_FIXTURE_DIR)


def _golden_fixture_bin_path():
    return os.path.join(_golden_fixture_dir(), _GOLDEN_FIXTURE_BIN)


def _golden_fixture_png_path():
    return os.path.join(_golden_fixture_dir(), _GOLDEN_FIXTURE_PNG)


def _make_checkerboard_image(side=8, channels=3, dtype=np.uint8):
    """Builds a deterministic ``(side, side, channels)`` checkerboard image."""
    img = np.zeros((side, side, channels), dtype=dtype)
    for y in range(side):
        for x in range(side):
            if (x + y) % 2 == 0:
                img[y, x, :] = 255
    return img


class TestImageIO:
    """PNG / JPEG image io round-trip tests."""

    def test_image_format_normalization(self):
        # 'jpg' canonicalizes to 'jpeg' and produces JPEG bytes.
        img = _make_checkerboard_image()
        as_jpg = to_binary(img, image_format='jpg')
        as_jpeg = to_binary(img, image_format='jpeg')
        # JPEG encoder is deterministic for a fixed input -> identical bytes.
        assert as_jpg == as_jpeg

        # Mixed case is accepted.
        as_PNG = to_binary(img, image_format='PNG')
        as_png = to_binary(img, image_format='png')
        assert as_PNG == as_png

        # The enum directly is also accepted.
        as_enum = to_binary(img, image_format=ImageFormat.PNG)
        assert as_enum == as_png

        # Unknown format is rejected with a helpful ValueError.
        with pytest.raises(ValueError, match='image_format'):
            to_binary(img, image_format='webp')
        with pytest.raises(ValueError, match='image_format'):
            to_binary(img, image_format='zip')

    def test_png_roundtrip_uint8_hwc(self):
        rng = np.random.default_rng(seed=0)
        img = rng.integers(0, 256, size=(13, 17, 3), dtype=np.uint8)
        encoded = to_binary(img, image_format=ImageFormat.PNG)
        decoded = from_binary(encoded)
        assert isinstance(decoded, torch.Tensor)
        assert decoded.dtype == torch.uint8
        assert tuple(decoded.shape) == (13, 17, 3)
        assert torch.equal(decoded, torch.from_numpy(img))

    def test_png_roundtrip_uint8_chw(self):
        rng = np.random.default_rng(seed=1)
        img_hwc = rng.integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
        img_chw = np.transpose(img_hwc, (2, 0, 1))  # (C, H, W)
        encoded = to_binary(img_chw, image_format=ImageFormat.PNG)
        decoded = from_binary(encoded)
        # Decoded back as HWC.
        assert tuple(decoded.shape) == (8, 12, 3)
        assert torch.equal(decoded, torch.from_numpy(img_hwc))

    def test_png_roundtrip_grayscale(self):
        rng = np.random.default_rng(seed=2)
        # (H, W, 1)
        gray3 = rng.integers(0, 256, size=(7, 11, 1), dtype=np.uint8)
        decoded3 = from_binary(to_binary(gray3, image_format=ImageFormat.PNG))
        assert tuple(decoded3.shape) == (7, 11, 1)
        assert torch.equal(decoded3, torch.from_numpy(gray3))

        # (H, W) uint8 — plain 2D grayscale.
        gray2 = rng.integers(0, 256, size=(9, 6), dtype=np.uint8)
        decoded2 = from_binary(to_binary(gray2, image_format=ImageFormat.PNG))
        # Decoded back as HWC with channels=1.
        assert tuple(decoded2.shape) == (9, 6, 1)
        assert torch.equal(decoded2.squeeze(-1), torch.from_numpy(gray2))

    def test_png_roundtrip_rgba(self):
        rng = np.random.default_rng(seed=3)
        img = rng.integers(0, 256, size=(8, 8, 4), dtype=np.uint8)
        decoded = from_binary(to_binary(img, image_format=ImageFormat.PNG))
        assert tuple(decoded.shape) == (8, 8, 4)
        assert torch.equal(decoded, torch.from_numpy(img))

    def test_jpeg_roundtrip_uint8_hwc(self):
        # Smooth gradient compresses well and gives a small MAE.
        h, w = 32, 32
        ramp = np.linspace(0, 255, w, dtype=np.float32)
        img = np.stack([np.tile(ramp, (h, 1))] * 3, axis=-1).astype(np.uint8)
        encoded = to_binary(img, image_format=ImageFormat.JPEG)
        decoded = from_binary(encoded)
        assert tuple(decoded.shape) == (h, w, 3)
        mae = (decoded.float() - torch.from_numpy(img).float()).abs().mean().item()
        assert mae < 4, f'JPEG MAE too high for smooth image: {mae}'

    def test_jpeg_skips_rgba(self):
        # 4-channel uint8 with image_format='jpeg' must NOT be encoded as
        # JPEG (JPEG accepts only 1 or 3 channels). It falls through to the
        # raw UINT8 path; the wire bytes therefore match the RAW baseline.
        rng = np.random.default_rng(seed=4)
        rgba = rng.integers(0, 256, size=(8, 8, 4), dtype=np.uint8)
        baseline = to_binary(rgba, image_format=ImageFormat.RAW)
        encoded = to_binary(rgba, image_format=ImageFormat.JPEG)
        assert encoded == baseline, (
            'JPEG must not engage on 4-channel uint8; expected RAW-baseline bytes'
        )
        decoded = from_binary(encoded)
        # Round-trip lands a torch tensor of the original shape (HWC) and
        # uint8 dtype, byte-identical to the input.
        assert tuple(decoded.shape) == (8, 8, 4)
        assert torch.equal(decoded, torch.from_numpy(rgba))

    def test_float_image_not_encoded(self):
        # Float image must NOT be detected as image-encodable; bytes match
        # the raw float32 baseline.
        rng = np.random.default_rng(seed=5)
        flt = rng.random(size=(8, 12, 3)).astype(np.float32)
        baseline = to_binary(flt, image_format=ImageFormat.RAW)
        encoded = to_binary(flt, image_format=ImageFormat.PNG)
        assert encoded == baseline, (
            'Float image must not engage PNG; expected RAW-baseline bytes'
        )
        decoded = from_binary(encoded)
        # Round-trip preserves dtype/shape/values exactly.
        assert decoded.dtype == torch.float32
        assert tuple(decoded.shape) == (8, 12, 3)
        assert torch.allclose(decoded, torch.from_numpy(flt))

    def test_image_in_dict(self):
        rng = np.random.default_rng(seed=6)
        img = rng.integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
        msg = {'pose': torch.eye(4, dtype=torch.float32),
               'img': img,
               'tag': 'render'}
        encoded = to_binary(msg, image_format=ImageFormat.PNG)
        decoded = from_binary(encoded)

        # Non-image entries unchanged.
        assert torch.allclose(decoded['pose'], msg['pose'])
        assert decoded['tag'] == 'render'
        # Image roundtrips losslessly.
        assert torch.equal(decoded['img'], torch.from_numpy(img))

        # Reseat as raw -> only the image bytes change; non-image entries
        # are byte-identical via independent encoding.
        baseline_no_img = to_binary({'pose': msg['pose'], 'tag': msg['tag']})
        baseline_no_img_png = to_binary({'pose': msg['pose'], 'tag': msg['tag']},
                                        image_format=ImageFormat.PNG)
        assert baseline_no_img == baseline_no_img_png

    def test_non_image_unchanged(self):
        # 1-D tensor must not be detected as image.
        v = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
        assert to_binary(v, image_format=ImageFormat.PNG) == to_binary(v)
        assert to_binary(v, image_format=ImageFormat.JPEG) == to_binary(v)

        # 2D float matrix must not auto-detect as grayscale image
        # (only uint8 grayscale is recognized).
        m = torch.zeros((3, 5), dtype=torch.float32)
        assert to_binary(m, image_format=ImageFormat.PNG) == to_binary(m)

        # 3D tensor with no channel-shaped dim.
        t = torch.zeros((5, 7, 9), dtype=torch.float32)
        assert to_binary(t, image_format=ImageFormat.PNG) == to_binary(t)

        # 3D tensor with degenerate spatial dims is not classified as image.
        d = torch.zeros((1, 1, 2), dtype=torch.float32)
        assert to_binary(d, image_format=ImageFormat.PNG) == to_binary(d)

        # 4D tensor (batch dim) is never an image.
        batched = torch.zeros((4, 8, 8, 3), dtype=torch.float32)
        assert to_binary(batched, image_format=ImageFormat.PNG) == to_binary(batched)

    def test_message_image_format_propagates(self):
        # A high-redundancy image (uniform color) compresses very well in PNG.
        side = 256
        solid = np.full((side, side, 3), fill_value=128, dtype=np.uint8)
        raw = encode_message('render', {'image': solid}, image_format=ImageFormat.RAW)
        png = encode_message('render', {'image': solid}, image_format=ImageFormat.PNG)
        assert len(png) * 4 < len(raw), \
            f"PNG ({len(png)} B) should be much smaller than raw ({len(raw)} B)"

        # Decoded message round-trips losslessly.
        decoded = from_binary(png)
        assert decoded[MESSAGE_TAG_KEY] == 'render'
        assert torch.equal(decoded[MESSAGE_CONTENT_KEY]['image'], torch.from_numpy(solid))

    def test_jpeg_quality_reduces_size(self):
        # Lower JPEG quality must yield smaller bytes, even for images nested
        # inside a dict (verifies jpeg_quality is threaded through the encoder).
        rng = np.random.default_rng(seed=7)
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        nested = {'outer': {'a': img, 'b': img}}
        high_q = encode_message('render', nested, image_format=ImageFormat.JPEG, jpeg_quality=95)
        low_q = encode_message('render', nested, image_format=ImageFormat.JPEG, jpeg_quality=10)
        assert len(low_q) < len(high_q), \
            f'Lower JPEG quality should be smaller: q10={len(low_q)} B, q95={len(high_q)} B'

    def test_torchvision_missing_message(self, monkeypatch):
        import torchvision.io as tvio

        def _raise(*_args, **_kwargs):
            raise ImportError('simulated missing torchvision')

        monkeypatch.setattr(tvio, 'encode_png', _raise)
        img = _make_checkerboard_image()
        # The encode helper re-raises with a torchvision-flavoured ImportError.
        with pytest.raises(ImportError, match='simulated missing torchvision'):
            to_binary(img, image_format=ImageFormat.PNG)

    def test_golden_fixture_python_side(self):
        path = _golden_fixture_bin_path()
        if not os.path.exists(path):
            pytest.skip(
                f'Golden fixture {path} not found. Generate via: '
                f'`python tests/python/kaolin/visualize/web/test_io.py`'
            )
        with open(path, 'rb') as f:
            encoded = f.read()
        decoded = from_binary(encoded)
        expected = _make_checkerboard_image()
        assert isinstance(decoded, torch.Tensor)
        assert tuple(decoded.shape) == expected.shape
        assert torch.equal(decoded, torch.from_numpy(expected))


def _generate_golden_fixture():
    """Writes the cross-language checkerboard fixtures under
    ``tests/samples/visualize/``:

    - ``checkerboard.png.bin`` — the binary I/O envelope around the PNG
      payload, used by both Python and TS golden-fixture tests.
    - ``checkerboard.png`` — the same image as a regular PNG so it can
      be inspected with any image viewer.

    Run via::

        python tests/python/kaolin/visualize/web/test_io.py
    """
    out_dir = _golden_fixture_dir()
    os.makedirs(out_dir, exist_ok=True)

    img = _make_checkerboard_image()
    encoded = to_binary(img, image_format=ImageFormat.PNG)
    bin_path = _golden_fixture_bin_path()
    with open(bin_path, 'wb') as f:
        f.write(encoded)
    print(f'Wrote {len(encoded)} bytes to {bin_path}')

    # Also drop a plain .png next to the .bin so it can be opened with a
    # regular image viewer / shipped to the JS side as a reference asset.
    from PIL import Image
    png_path = _golden_fixture_png_path()
    Image.fromarray(img, mode='RGB').save(png_path, format='PNG')
    print(f'Wrote PNG to {png_path}')


if __name__ == '__main__':
    _generate_golden_fixture()
