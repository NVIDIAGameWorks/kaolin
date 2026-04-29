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

import collections
import json
import logging
import numpy as np
import torch
from enum import Enum, IntEnum


logger = logging.getLogger(__name__)

class BinaryIoDataType(IntEnum):
    """Types supported for binary serialization by the
    :func:`to_binary` and :func:`from_binary` functions.
    Matches ``window.kaolin.io.BinaryIoDataType`` enum in
    :doc:`Kaolin Javascript API <kaolin.visualize.dash.javascript>` for client-server compatibility.
    """
    INT8 = 0  #: Signed 8-bit integer, corresponds to torch.int8
    UINT8 = 1  #: Unsigned 8-bit integer, corresponds to torch.uint8
    INT16 = 2  #: Signed 16-bit integer, corresponds to torch.int16
    INT32 = 3  #: Signed 32-bit integer, corresponds to torch.int32 and used for regular ints
    UINT32 = 4  #: Unsigned 32-bit integer, corresponds to torch.uint32
    INT64 = 5  #: Signed 64-bit integer, corresponds to torch.int64
    FLOAT16 = 6  #: 16-bit floating point, corresponds to torch.float16
    FLOAT32 = 7  #: 32-bit floating point, corresponds to torch.float32 and used for regular floats
    FLOAT64 = 8  #: 64-bit floating point, corresponds to torch.float64
    STRING = 9  #: UTF-8 encoded string
    DICT = 10  #: Dictionary/Map of named values
    LIST = 11  #: List of values
    PNG = 12  #: PNG-compressed image; payload is the encoded PNG byte stream
    JPEG = 13  #: JPEG-compressed image; payload is the encoded JPEG byte stream
    UNSUPPORTED = 100  #: Unsupported type marker


class ImageFormat(Enum):
    """Image-encoding mode threaded through :func:`encode_message` and :func:`to_binary`.

    Selects how image-shaped uint8 tensors are encoded:

    - :attr:`RAW` — no image detection; tensors are written as raw typed
      arrays (default; bytes are unchanged from the pre-PNG/JPEG format).
    - :attr:`PNG` — image-shaped uint8 tensors with 1, 3, or 4 channels are
      compressed as PNG.
    - :attr:`JPEG` — image-shaped uint8 tensors with 1 or 3 channels are
      compressed as JPEG. 4-channel tensors fall through to the raw uint8
      path (JPEG cannot express alpha).
    """
    RAW = 'raw'
    PNG = 'png'
    JPEG = 'jpeg'

    @classmethod
    def coerce(cls, value):
        """Accepts an :class:`ImageFormat`, ``None``, or a string ``'raw'``,
        ``'png'``, ``'jpeg'``, ``'jpg'`` (case-insensitive). Returns the
        canonical enum member or raises ``ValueError``."""
        if value is None:
            return cls.RAW
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            canonical = value.lower()
            if canonical == 'jpg':
                canonical = 'jpeg'
            for member in cls:
                if member.value == canonical:
                    return member
        raise ValueError(
            f"image_format must be one of 'raw', 'png', 'jpeg', 'jpg' "
            f"(or an ImageFormat enum member); got {value!r}")


__np_type_mappings =  [(BinaryIoDataType.INT8, np.dtype(np.int8)),
                       (BinaryIoDataType.UINT8, np.dtype(np.uint8)),
                       (BinaryIoDataType.INT16, np.dtype(np.int16)),
                       (BinaryIoDataType.INT32, np.dtype(np.int32)),
                       (BinaryIoDataType.UINT32, np.dtype(np.uint32)),
                       (BinaryIoDataType.INT64, np.dtype(np.int64)),
                       (BinaryIoDataType.FLOAT16, np.dtype(np.float16)),
                       (BinaryIoDataType.FLOAT32, np.dtype(np.float32)),
                       (BinaryIoDataType.FLOAT64, np.dtype(np.float64))]
__np_type_to_bytes = {np.dtype(np.int8): 1,
                      np.dtype(np.uint8): 1,
                      np.dtype(np.int16): 2,
                      np.dtype(np.int32): 4,
                      np.dtype(np.uint32): 4,
                      np.dtype(np.int64): 8,
                      np.dtype(np.float16): 2,
                      np.dtype(np.float32): 4,
                      np.dtype(np.float64): 8}
__io_data_type_to_np = dict(__np_type_mappings)
__np_to_io_data_type = dict([(x[1], x[0]) for x in __np_type_mappings])

MESSAGE_TAG_KEY = 'tag'
MESSAGE_CONTENT_KEY = 'msg'

# Channel counts each codec accepts. JPEG cannot express alpha, so a 4-channel
# uint8 image is intentionally NOT detected as JPEG-encodable; it falls back
# to the raw UINT8 wire format.
_PNG_ALLOWED_CHANNELS = (1, 3, 4)
_JPEG_ALLOWED_CHANNELS = (1, 3)
_DEFAULT_JPEG_QUALITY = 90
_DEFAULT_IMAGE_FORMAT = ImageFormat.PNG


def encode_message(tag, content, binary=True, image_format=_DEFAULT_IMAGE_FORMAT, jpeg_quality=_DEFAULT_JPEG_QUALITY):
    """Encodes a tagged message using the standard ``tag`` / ``msg`` keys.

    Args:
        tag: tag or message type to set.
        content: content to set.
        binary: if True, will be encoded as binary and as JSON otherwise.
        image_format: :class:`ImageFormat` (or its string alias ``'raw'`` /
            ``'png'`` / ``'jpeg'`` / ``'jpg'``). Defaults to
            :data:`_DEFAULT_IMAGE_FORMAT`. Forwarded to :func:`to_binary`; only
            relevant for binary encoding.
        jpeg_quality: 0..100 quality for JPEG encoding (default
            :data:`_DEFAULT_JPEG_QUALITY`). Forwarded to :func:`to_binary`;
            only used when ``image_format`` is JPEG.

    Returns:
        bytes (binary=True) or JSON string (binary=False).
    """
    msg = {MESSAGE_TAG_KEY: tag, MESSAGE_CONTENT_KEY: content}
    if binary:
        return to_binary(msg, image_format=image_format, jpeg_quality=jpeg_quality)
    else:
        return json.dumps(msg)


def to_binary(value, image_format=_DEFAULT_IMAGE_FORMAT, jpeg_quality=_DEFAULT_JPEG_QUALITY):
    """Encodes any message as binary.

    Image-encodable detection is **uint8-only**: only ``np.ndarray`` /
    ``torch.Tensor`` of dtype ``uint8`` and a recognized image layout
    (``(H, W)``, ``(H, W, C)``, ``(C, H, W)``) are routed through the
    PNG / JPEG codec when ``image_format`` requests it. Float tensors,
    integer tensors of other dtypes, batched tensors, and tensors with
    channel counts the requested codec cannot express are written as
    raw typed arrays — i.e. unchanged from the ``'raw'`` baseline.

    Args:
        value: value to encode.
        image_format: :class:`ImageFormat` or its string alias
            (``'raw'`` / ``'png'`` / ``'jpeg'`` / ``'jpg'``). Defaults to
            :data:`_DEFAULT_IMAGE_FORMAT`. ``'jpg'`` is normalized to
            :attr:`ImageFormat.JPEG`.
        jpeg_quality: 0..100 quality for JPEG encoding (default
            :data:`_DEFAULT_JPEG_QUALITY`); only used when ``image_format``
            is JPEG.

    Returns:
        Encoded ``bytes``.
    """
    return value_to_binary(value, 0, image_format=image_format, jpeg_quality=jpeg_quality)


def from_binary(bytes_msg: bytes):
    """
    Decodes value from binary.

    Args:
        bytes_msg:

    Returns:

    """
    res, read_bytes = value_from_binary(bytes_msg, 0)
    if read_bytes != len(bytes_msg):
        logger.warning(f'Read {read_bytes}, not full message length {len(bytes_msg)}')
    return res


def np_type_from_type_id(type_id):
    """
    Returns NumPy dtype and bytes per element for given type ID.

    Args:
        type_id: Binary I/O data type ID

    Returns:
        tuple: (numpy_dtype, bytes_per_element) or (None, 0) if unknown
    """
    np_type = __io_data_type_to_np.get(type_id, None)
    if np_type is not None:
        num_bytes = __np_type_to_bytes.get(np_type, 0)
    else:
        num_bytes = 0

    return np_type, num_bytes


def value_to_type(converted_value, image_format):
    """Maps an already-:func:`convert_value_to_supported_format`-ed value to
    its :class:`BinaryIoDataType`.

    ``image_format`` selects whether image-shaped uint8 ndarrays are
    reported as :attr:`BinaryIoDataType.PNG` / :attr:`~BinaryIoDataType.JPEG`
    (and therefore routed through the codec by :func:`value_to_binary`),
    or as plain :attr:`~BinaryIoDataType.UINT8` arrays.
    """
    image_format = ImageFormat.coerce(image_format)
    if isinstance(converted_value, str):
        return BinaryIoDataType.STRING
    elif isinstance(converted_value, collections.abc.Mapping):
        return BinaryIoDataType.DICT
    elif isinstance(converted_value, list):
        return BinaryIoDataType.LIST
    elif isinstance(converted_value, np.ndarray):
        if image_format == ImageFormat.PNG:
            is_image, _ = _is_image_like(converted_value, allowed_channels=_PNG_ALLOWED_CHANNELS)
            if is_image:
                return BinaryIoDataType.PNG
        elif image_format == ImageFormat.JPEG:
            is_image, _ = _is_image_like(converted_value, allowed_channels=_JPEG_ALLOWED_CHANNELS)
            if is_image:
                return BinaryIoDataType.JPEG
        return __np_to_io_data_type.get(converted_value.dtype, BinaryIoDataType.UNSUPPORTED)
    else:
        return BinaryIoDataType.UNSUPPORTED


def convert_value_to_supported_format(value):
    if isinstance(value, str):
        return value
    elif torch.is_tensor(value):
        return convert_value_to_supported_format(value.detach().cpu().numpy())
    elif isinstance(value, collections.abc.Mapping):
        return value
    elif isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return value.as_type(np.int32)
        # TODO: possibly convert other types
        return value
    elif isinstance(value, list):
        if len(value) == 0:
            return value  # Keep empty lists as lists

        # Check if all elements are numbers (int or float)
        all_numbers = all(isinstance(item, (int, float, bool)) for item in value)

        if all_numbers:
            # Check if all are integers
            all_integers = all(isinstance(item, int) or isinstance(item, bool) for item in value)
            if all_integers:
                return np.array(value, dtype=np.int32)
            else:
                return np.array(value, dtype=np.float32)
        else:
            # Mixed types - keep as list for LIST support
            return value
    elif isinstance(value, int) or isinstance(value, bool):
        return np.array([value], dtype=np.int32)
    elif isinstance(value, float):
        return np.array([value], dtype=np.float32)
    else:
        raise ValueError(f'Cannot encode value of type {type(value)} to binary')


def _is_image_like(value, *, allowed_channels):
    """Detects whether ``value`` is an image-shaped uint8 ``np.ndarray``.

    Image detection is **uint8-only**: float / non-uint8 ndarrays return
    ``(False, None)`` so they always fall through to the raw typed-array
    path. Recognized layouts:

    - ``(H, W)`` uint8 with ``min(H, W) >= 2`` -> ``'hw'`` (single-channel
      grayscale).
    - ``(H, W, C)`` uint8 with ``C in allowed_channels`` and
      ``min(H, W) >= 2`` -> ``'hwc'``.
    - ``(C, H, W)`` uint8 with ``C in allowed_channels`` and
      ``min(H, W) >= 2`` -> ``'chw'``.

    When both the first and last dim are valid channel counts the function
    prefers HWC, matching the kaolin / dash convention.

    Args:
        value: the candidate ``np.ndarray``. Always coming from
            :func:`convert_value_to_supported_format`, so torch / list /
            primitive shapes need not be considered here.
        allowed_channels: iterable of channel counts that the eventual
            codec accepts. **Required**: callers must pass the codec's
            channel set explicitly so detection stays in lock-step with
            encoding.

    Returns:
        tuple ``(is_image, layout)`` where ``layout`` is ``'hw'``, ``'hwc'``,
        ``'chw'`` or ``None``.
    """
    if value.dtype != np.uint8:
        return False, None

    shape = tuple(value.shape)

    # Spatial dims smaller than this are too degenerate to plausibly be images;
    # this prevents tensors like (1, 1, 2) from being mis-classified.
    _MIN_SPATIAL = 2

    if len(shape) == 2:
        h, w = shape
        if min(h, w) < _MIN_SPATIAL:
            return False, None
        return True, 'hw'
    if len(shape) != 3:
        return False, None

    last_ok = shape[-1] in allowed_channels
    first_ok = shape[0] in allowed_channels
    if last_ok and not first_ok:
        h, w = shape[0], shape[1]
        if min(h, w) < _MIN_SPATIAL:
            return False, None
        return True, 'hwc'
    if first_ok and not last_ok:
        h, w = shape[1], shape[2]
        if min(h, w) < _MIN_SPATIAL:
            return False, None
        return True, 'chw'
    if last_ok and first_ok:
        # Ambiguous: fall back to HWC, which matches kaolin/dash conventions.
        h, w = shape[0], shape[1]
        if min(h, w) < _MIN_SPATIAL:
            return False, None
        return True, 'hwc'
    return False, None


def _to_chw_uint8(value):
    """Convert an image-detected uint8 ndarray into a contiguous
    ``(C, H, W)`` uint8 ``torch.Tensor``.

    Always called after :func:`convert_value_to_supported_format` and
    :func:`_is_image_like` (with the codec's channel set) returned True,
    so the input is guaranteed to be a ``np.ndarray`` of dtype ``uint8``
    with a recognized image layout. No alpha drop, no float-to-uint8
    conversion: anything that the codec cannot represent must have been
    rejected upstream by :func:`_is_image_like`.
    """
    is_image, layout = _is_image_like(value, allowed_channels=_PNG_ALLOWED_CHANNELS)
    assert is_image, (
        f'_to_chw_uint8 called on non-image-like value with shape {tuple(value.shape)} '
        f'and dtype {value.dtype}; this is a bug in the caller.')

    tensor = torch.from_numpy(value)
    if layout == 'hw':
        return tensor.unsqueeze(0).contiguous()
    if layout == 'hwc':
        return tensor.permute(2, 0, 1).contiguous()
    # 'chw'
    return tensor.contiguous()


def _encode_png_rgba_pillow(tensor_chw_u8):
    """Pillow-based PNG encoder for the RGBA case (torchvision's
    ``encode_png`` rejects 4-channel input as of v0.20+)."""
    from PIL import Image
    import io as _stdlib_io
    arr = tensor_chw_u8.permute(1, 2, 0).contiguous().cpu().numpy()
    img = Image.fromarray(arr, mode='RGBA')
    buf = _stdlib_io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def _encode_image_bytes(tensor_chw_u8, fmt, jpeg_quality=_DEFAULT_JPEG_QUALITY):
    """Encode a CHW uint8 tensor as PNG or JPEG bytes via ``torchvision.io``.
    Falls back to Pillow for the 4-channel PNG case, which torchvision's
    native ``encode_png`` does not support.
    """
    try:
        import torchvision.io as tvio
    except ImportError as e:
        raise ImportError(
            'torchvision is required for PNG/JPEG image io. '
            'Install with: pip install torchvision'
        ) from e

    channels = tensor_chw_u8.shape[0]

    if fmt == ImageFormat.PNG:
        if channels == 4:
            return _encode_png_rgba_pillow(tensor_chw_u8)
        encoded = tvio.encode_png(tensor_chw_u8)
    elif fmt == ImageFormat.JPEG:
        encoded = tvio.encode_jpeg(tensor_chw_u8, quality=jpeg_quality)
    else:
        raise ValueError(f"_encode_image_bytes: unknown fmt {fmt!r}")

    if torch.is_tensor(encoded):
        return bytes(encoded.cpu().numpy())
    return bytes(encoded)


def _decode_image_bytes(payload, type_code):
    """Decode PNG/JPEG byte payload to an HWC uint8 ``torch.Tensor``."""
    try:
        import torchvision.io as tvio
    except ImportError as e:
        raise ImportError(
            'torchvision is required for PNG/JPEG image io. '
            'Install with: pip install torchvision'
        ) from e

    payload_tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    img_chw = tvio.decode_image(payload_tensor)
    return img_chw.permute(1, 2, 0).contiguous()

def gap_until_offset_n(current_offset: int, n: int) -> int:
    """
    Calculate the gap needed to align current_offset to a multiple of n.
    Arrays such as Int32Array cannot start at offsets that are not
    a multiple of 4. This helps us find the right offset.

    Args:
        current_offset: Current byte offset
        n: Alignment requirement (e.g., 4 for 4-byte alignment)

    Returns:
        Number of bytes to add to reach proper alignment
    """
    return (n - (current_offset % n)) % n


def gap_until_offset_4(current_offset: int) -> int:
    return gap_until_offset_n(current_offset, 4)


def string_from_binary(bytes_msg, offset, byte_length):
    """
    Decodes UTF-8 string of specified length from binary data.

    Args:
        bytes_msg: Input binary data (bytes)
        offset: Byte offset in data
        byte_length: String length in bytes

    Returns:
        Decoded string
    """
    if byte_length == 0:
        return ''
    string_bytes = bytes_msg[offset:offset + byte_length]
    return string_bytes.decode('utf-8')


def string_to_binary(string):
    """
    Encodes string to binary data using UTF-8 encoding.

    Args:
        string: String to encode

    Returns:
        bytes containing UTF-8 encoded bytes
    """
    return string.encode('utf-8')


def typed_value_from_binary(bytes_msg, offset, length, type_code):
    # Length is byte length for strings/PNG/JPEG, but num elements for other types
    if type_code == BinaryIoDataType.STRING:
        return string_from_binary(bytes_msg, offset, length), length
    elif type_code == BinaryIoDataType.DICT:
        value, read_bytes = _dict_from_binary(bytes_msg, length=length, offset=offset)
        return value, read_bytes
    elif type_code == BinaryIoDataType.LIST:
        value, read_bytes = _list_from_binary(bytes_msg, length=length, offset=offset)
        return value, read_bytes
    elif type_code in (BinaryIoDataType.PNG, BinaryIoDataType.JPEG):
        payload = bytes(bytes_msg[offset:offset + length])
        img_hwc = _decode_image_bytes(payload, type_code)
        return img_hwc, length
    else:
        np_type, bytes_per_element = np_type_from_type_id(type_code)
        if np_type is None:
            return None, 0
        read_bytes = gap_until_offset_n(offset, bytes_per_element)
        value = np.frombuffer(bytes_msg, dtype=np_type, count=length, offset=offset + read_bytes)
        read_bytes += length * bytes_per_element
        return value, read_bytes


def value_from_binary(bytes_msg, offset):
    read_bytes = gap_until_offset_4(offset)

    metadata_length = 2
    metadata = np.frombuffer(bytes_msg, dtype=np.int32, count=metadata_length, offset=offset + read_bytes)
    read_bytes += metadata_length * 4
    shape_length = metadata[0]
    type_code = metadata[1]

    is_primitive = False
    if shape_length > 0:
        shape = np.frombuffer(bytes_msg, dtype=np.int32, count=shape_length, offset=offset + read_bytes)
        read_bytes += shape_length * 4
        length = np.prod(shape)
    else:
        length = 1
        is_primitive = True
    value, value_read_bytes = typed_value_from_binary(bytes_msg, offset + read_bytes, length, type_code)
    read_bytes += value_read_bytes
    if type_code in (BinaryIoDataType.PNG, BinaryIoDataType.JPEG):
        # Already a torch tensor of correct (H, W, C) shape; the wire 'shape'
        # carries the byte length of the compressed payload, not image dims.
        return value, read_bytes
    if isinstance(value, np.ndarray):
        if is_primitive:
            value = value[0].item()
        else:
            try:
                # TODO: this array is not writable; figure out what the behavior should be
                value = torch.from_numpy(value).reshape([x for x in shape])  # return in torch, as that is Kaolin i/o convention
            except ValueError as e:
                logger.error(f'Decoded shape does not match value size {e}')
    return value, read_bytes


def value_to_binary(in_value, initial_offset=0, image_format=_DEFAULT_IMAGE_FORMAT,
                    jpeg_quality=_DEFAULT_JPEG_QUALITY):
    """Encodes a single value to binary, prepending the wire metadata header.

    Image-encodable detection is **uint8-only**: only ``np.ndarray`` /
    ``torch.Tensor`` of dtype ``uint8`` are considered for PNG / JPEG
    encoding when ``image_format`` requests it. Every other path
    (floats, non-uint8 ints, tensors with shapes that don't look like an
    image, JPEG-incompatible 4-channel inputs) is byte-identical to the
    ``'raw'`` baseline.
    """
    is_primitive_number = isinstance(in_value, int) or isinstance(in_value, float) or isinstance(in_value, bool)
    value = convert_value_to_supported_format(in_value)

    type_code = value_to_type(value, image_format=image_format)
    if type_code == BinaryIoDataType.UNSUPPORTED:
        raise ValueError(f'Cannot encode value of type {type(value)}')

    # Insert alignment
    result = bytes(gap_until_offset_4(initial_offset))

    if type_code == BinaryIoDataType.STRING:
        encoded_value = string_to_binary(value)
        shape = np.array([len(encoded_value)], dtype=np.int32)
        bytes_per_elem = 1
    elif type_code == BinaryIoDataType.DICT:
        shape = np.array([len(value)], dtype=np.int32)
        encoded_value = _dict_to_binary(value, initial_offset=initial_offset + len(result) + 3 * 4,
                                        image_format=image_format, jpeg_quality=jpeg_quality)
        bytes_per_elem = 1  # alignment is already accounted for
    elif type_code == BinaryIoDataType.LIST:
        shape = np.array([len(value)], dtype=np.int32)
        encoded_value = _list_to_binary(value, initial_offset=initial_offset + len(result) + 3 * 4,
                                        image_format=image_format, jpeg_quality=jpeg_quality)
        bytes_per_elem = 1  # alignment is already accounted for
    elif type_code in (BinaryIoDataType.PNG, BinaryIoDataType.JPEG):
        fmt = ImageFormat.PNG if type_code == BinaryIoDataType.PNG else ImageFormat.JPEG
        chw = _to_chw_uint8(value)
        encoded_value = _encode_image_bytes(chw, fmt=fmt, jpeg_quality=jpeg_quality)
        shape = np.array([len(encoded_value)], dtype=np.int32)
        bytes_per_elem = 1
    else:
        encoded_value = value.tobytes()
        # set shape len to 0 for primitives
        shape = np.array([] if is_primitive_number else value.shape, dtype=np.int32)
        bytes_per_elem = __np_type_to_bytes.get(value.dtype, 1)

    # Encode metadata and shape
    result += np.array([len(shape), type_code], dtype=np.int32).tobytes() + shape.tobytes()

    # Insert alignment
    extra_bytes = gap_until_offset_n(len(result) + initial_offset, bytes_per_elem)
    result += bytes(extra_bytes)
    result += encoded_value
    return result


def named_value_from_binary(bytes_msg, offset):
    read_bytes = gap_until_offset_4(offset)
    name_length = np.frombuffer(bytes_msg, dtype=np.int32, count=1, offset=offset + read_bytes)[0]  # in bytes
    read_bytes += 4
    name = string_from_binary(bytes_msg, offset + read_bytes, name_length)
    read_bytes += name_length
    value, value_read_bytes = value_from_binary(bytes_msg, offset + read_bytes)
    read_bytes += value_read_bytes
    return name, value, read_bytes


def named_value_to_binary(name, value, initial_offset=0, image_format=_DEFAULT_IMAGE_FORMAT,
                          jpeg_quality=_DEFAULT_JPEG_QUALITY):
    # We assume offset is appropriate for int32
    bin_str = string_to_binary(name)
    result = bytes(gap_until_offset_4(initial_offset))
    result += int32_to_binary(len(bin_str))
    result += bin_str
    result += value_to_binary(value, initial_offset=initial_offset + len(result),
                              image_format=image_format, jpeg_quality=jpeg_quality)
    return result


def _dict_from_binary(bytes_msg, length, offset=0):
    """Converts bytes message to dictionary.
    Must be compatible with: nvidia.Controller.prototype.encodeDrawingRequest.

    Args:
        @param bytes_msg: raw bytes to decode
        @param offset: start read offset in bytes
        @param length: number of key-value pairs

    Return:
        metadata dict, total_read_bytes
    """
    total_read_bytes = 0
    res = {}
    for i in range(length):
        name, value, read_bytes = named_value_from_binary(bytes_msg, offset + total_read_bytes)
        res[name] = value
        total_read_bytes += read_bytes

    return res, total_read_bytes


def _list_from_binary(bytes_msg, length, offset=0):
    """Converts bytes message to list.

    Args:
        @param bytes_msg: raw bytes to decode
        @param offset: start read offset in bytes
        @param length: number of elements

    Return:
        list, total_read_bytes
    """
    total_read_bytes = 0
    res = []
    for i in range(length):
        value, read_bytes = value_from_binary(bytes_msg, offset + total_read_bytes)
        res.append(value)
        total_read_bytes += read_bytes

    return res, total_read_bytes



def int32_to_binary(single_int):
    return np.array([single_int], dtype=np.int32).tobytes()


def _dict_to_binary(in_dict, initial_offset=0, image_format=_DEFAULT_IMAGE_FORMAT,
                    jpeg_quality=_DEFAULT_JPEG_QUALITY):
    result = bytes()

    for name, value in in_dict.items():
        result += named_value_to_binary(name, value, initial_offset=initial_offset + len(result),
                                        image_format=image_format, jpeg_quality=jpeg_quality)
    return result


def _list_to_binary(in_list, initial_offset=0, image_format=_DEFAULT_IMAGE_FORMAT,
                    jpeg_quality=_DEFAULT_JPEG_QUALITY):
    result = bytes()

    for value in in_list:
        result += value_to_binary(value, initial_offset=initial_offset + len(result),
                                  image_format=image_format, jpeg_quality=jpeg_quality)
    return result
