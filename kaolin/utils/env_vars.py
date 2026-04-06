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

from enum import Enum


__all__ = ['KaolinTestEnvVars', 'KaolinEnvVars']


class KaolinTestEnvVars(str, Enum):
    """Names of environment variables configuring Kaolin tests."""

    # TODO: replace hard-coded strings with these in all the tests --> we have too many hidden env vars controlling behavior
    # Test datasets and assets
    TEST_SCANNED_TOYS = 'KAOLIN_TEST_SCANNED_TOYS'
    TEST_GSPLATS_DIR = 'KAOLIN_TEST_GSPLATS_DIR'
    TEST_MODELNET_PATH = 'KAOLIN_TEST_MODELNET_PATH'
    TEST_SHAPENETV1_PATH = 'KAOLIN_TEST_SHAPENETV1_PATH'
    TEST_SHAPENETV2_PATH = 'KAOLIN_TEST_SHAPENETV2_PATH'
    TEST_SHREC16_PATH = 'KAOLIN_TEST_SHREC16_PATH'

    # Optional test backends
    TEST_NVDIFFRAST = 'KAOLIN_TEST_NVDIFFRAST'
    TEST_NVDIFFRAST_OPENGL = 'KAOLIN_TEST_NVDIFFRAST_OPENGL'


class KaolinEnvVars(str, Enum):
    """Names of environment variables read or documented by Kaolin."""

    #: Filesystem path to the root directory of the scanned toys dataset.
    #: Set ``KAOLIN_SCANNED_TOYS_PATH`` in the shell to override the default
    #: location exposed as :py:data:`~kaolin.utils.bundled_data.SCANNED_TOYS_PATH`
    #: (typically under ``sample_data`` in a source checkout).
    SCANNED_TOYS_PATH = 'KAOLIN_SCANNED_TOYS_PATH'
