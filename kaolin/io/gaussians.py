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

from . import ply

__all__ = [ 'import_gaussiancloud' ]

def import_gaussiancloud(filename: str):
    """ Automatically selects appropriate reader and reads 3D Gaussian Splat Cloud from file.
    Supported formats: USD and ply input. For more fine-grained control refer to
    :py:func:`~kaolin.io.usd.import_gaussiancloud`,
    :py:func:`~kaolin.io.ply.import_gaussiancloud`.

    Args:
        filename (str): path to the filename

    Returns:
        (GaussianSplatModel):
        A single gaussian cloud instance, or ``None`` if no gaussian clouds are found (USD only).
    """
    extension = filename.split('.')[-1].lower()
    if extension == 'ply':
        res = ply.import_gaussiancloud(filename)
    elif extension in ["usd", "usda", "usdc", "usdz"]:
        try:
            from . import usd
        except ImportError:
            raise ImportError("Cannot use usd import features, usd-core is not installed")
        res = usd.import_gaussiancloud(filename)
    else:
        raise ValueError(f'Unsupported Gaussian Splat filename extension {extension}')
    return res
