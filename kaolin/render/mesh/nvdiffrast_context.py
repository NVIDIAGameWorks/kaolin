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

from functools import partial
import logging

logger = logging.getLogger(__name__)

_device2glctx = {}
_has_nvdiffrast = False
_default_context_fn = lambda device: None

try:
    import nvdiffrast.torch as nvdiff
    _has_nvdiffrast = True
    _default_context_fn = partial(nvdiff.RasterizeGLContext, output_db=False)  # Faster than CUDA (~2x)
except ImportError:
    nvdiff = None
    logger.info("Cannot import nvdiffrast")


def _log_not_available():
    logger.warning(f'Nvdiffrast is not available; operation has no effect')


def nvdiffrast_is_available():
    """ Returns True if nvdiffrast is available, False otherwise."""
    return _has_nvdiffrast


def nvdiffrast_use_cuda():
    """ Configures nvdiffrast back end to use `nvdiffrast.torch.RasterizeCudaContext` by default."""
    global _default_context_fn
    if nvdiffrast_is_available():
        _default_context_fn = nvdiff.RasterizeCudaContext
    else:
        _log_not_available()


def nvdiffrast_use_opengl():
    """ Configures nvdiffrast back end to use `nvdiffrast.torch.RasterizeGLContext` by default."""
    global _default_context_fn
    if nvdiffrast_is_available():
        _default_context_fn = partial(nvdiff.RasterizeGLContext, output_db=False)
    else:
        _log_not_available()


def set_default_nvdiffrast_context(context, device="cuda"):
    """ Allows manually setting default nvdiffrast context to the given value for a specific device.

    Args:
        context (nvdiffrast.torch.RasterizeCudaContext or nvdiffrast.torch.RasterizeGLContext): context instance
        device (str, torch.device): pytorch device
    """
    if nvdiffrast_is_available():
        device_name = str(device)
        if device_name in _device2glctx:
            logger.warning(f'Replacing default nvdiffrast context for device {device_name}.')
        _device2glctx[device_name] = context
    else:
        _log_not_available()


def default_nvdiffrast_context(device, raise_error=False):
    """ Returns existing context for device, or creates one. To configure nvdiffrast to use opengl or CUDA back
    end by default call :func:`nvdiffrast_use_cuda` or func:`nvdiffrast_use_opengl`.

    Args:
        device (str or torch.device): device for the context

    Returns:
        nvdiffrast.torch.RasterizeCudaContext or nvdiffrast.torch.RasterizeGLContext
    """
    if not nvdiffrast_is_available():
        if raise_error:
            raise ValueError("nvdiffrast must be installed to be used as backend, but failed to import. "
                             "See https://nvlabs.github.io/nvdiffrast/#installation for installation instructions.")
        return None

    device_name = str(device)
    if device_name not in _device2glctx:
        _device2glctx[device_name] = _default_context_fn(device=device)
    return _device2glctx[device_name]

