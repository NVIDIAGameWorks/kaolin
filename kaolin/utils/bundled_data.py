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

"""Paths and constants for bundled Kaolin sample data and toy assets,
as well as download utilities for data stored in the cloud.
"""

import os
import wget
import hashlib
import zipfile
from kaolin.utils.env_vars import KaolinEnvVars

__all__ = ['BUNDLED_DATA_PATH', 'SAMPLE_MESHES_PATH', 'SCANNED_TOYS_PATH', 'SCANNED_TOYS_NAMES',
           'download_scanned_toys_dataset']

# TODO: add our meshes to manifest to make them available
#: Absolute path to the ``sample_data`` tree shipped with Kaolin (meshes, scanned toys, etc.).
#: In a source checkout, if the package lives at ``<repo_root>/kaolin``, this is
#: ``<repo_root>/sample_data`` (the concrete string is resolved at import time).
BUNDLED_DATA_PATH = os.path.realpath(os.path.join(
    os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir, 'sample_data'))  #: :meta hide-value:

#: Bundled mesh assets directory: ``sample_data/meshes`` next to the ``kaolin`` package
#: (same layout as ``BUNDLED_DATA_PATH`` / ``meshes``).
SAMPLE_MESHES_PATH = os.path.join(BUNDLED_DATA_PATH, 'meshes')  #: :meta hide-value:

#: Bundled scanned-toy assets directory: by default ``sample_data/scanned_toys`` relative to
#: the repo root in a source checkout, unless overridden with ``KAOLIN_SCANNED_TOYS_PATH``.
SCANNED_TOYS_PATH = os.getenv(KaolinEnvVars.SCANNED_TOYS_PATH) or os.path.join(BUNDLED_DATA_PATH, 'scanned_toys')  #: :meta hide-value:

#: Toy identifiers matching files in the cloud/bundled toys dataset (individual ``.usdc`` / ``.pt`` names, etc.).
#: The dataset also includes a combined ``BluehairRagdoll_multi.usdc`` (two Gaussian clouds: original and transformed)
#: and ``BluehairRagdoll_compressed.usdc`` (same clouds in float16, spherical harmonics degree 0).
SCANNED_TOYS_NAMES = ['BluehairRagdoll', 'bublik_octopus', 'knit_meow', 'mer_elephant', 'stink_raccoon', 'sunflower_baby']

# Expected MD5 hex digests for toy Gaussian ``.usdc`` files under ``SCANNED_TOYS_PATH``.
_TOYS_USDC_CHECKSUMS = {
    'BluehairRagdoll.usdc': '194fc1779ed4a2c3c310c3dfc4fb7063',
    'bublik_octopus.usdc': '87c678debbc4c1dbbbd109cf14a8af66',
    'knit_meow.usdc': '23a80128a84d9c489c5a5a1f20fc42e7',
    'mer_elephant.usdc': '220c8ae6c73efb93cabeddb13b9b759e',
    'stink_raccoon.usdc': '7b9a9c05f080e4526dac45d2d7faac11',
    'sunflower_baby.usdc': 'afb4d81f5c8c63ffd9778cfae608c138',
    'BluehairRagdoll_multi.usdc': '7ca53e1b359cc8ce04d272c132ea4fd8',
    'BluehairRagdoll_compressed.usdc': '8813ce84ab6a349a98aa922e79053b67'
}

# Expected MD5 hex digests for toy Gaussian ``.pt`` files under ``SCANNED_TOYS_PATH``.
_TOYS_PT_CHECKSUMS = {
    'BluehairRagdoll.pt': '7e84a773e5402a4cc8e84e8c8544ae0a',
    'bublik_octopus.pt': '7f34c1e5f0a7ea6634ea4a92b81835e7',
    'knit_meow.pt': 'b917ddd7485b4cf7e9414164e71e3707',
    'mer_elephant.pt': '3cee58ef3933166f5a7f60934362c3da',
    'stink_raccoon.pt': 'dfb63443a8c12649c16fe8f364caee86',
    'sunflower_baby.pt': 'b93bc44375d115aa5cebaf3c3c68cdec',
}

# TODO: gingerbread was probably put in by mistake: 'gingerbread.ply': '9fa27afbc0e26bd933e156bb48254ff8',
_TOYS_PLY_CHECKSUMS = {
    'BluehairRagdoll.ply': 'f3a6591e7dc497d2fdbd1bb654f01baf',
    'bublik_octopus.ply': '4bd216545854171ef69d5c62a0aea369',
    'knit_meow.ply': 'f413b1f5c3cc7ccbc724643749f06d57',
    'mer_elephant.ply': '613f87a424c9eba9ecafd8b91150330c',
    'stink_raccoon.ply': 'e78798c1d2b6b4262b6c05636022fa15',
    'sunflower_baby.ply': '5f464485ac524c09e3a9b8a9f7d2b7d5',
}

# TODO: document details, point to a whitepaper when available
def download_scanned_toys_dataset():
    """Downloads Scanned Toys Dataset, put together by the Kaolin team. """
    def _have_expected_files(file_to_checksum, return_reason=False):
        have_all = True
        msg = ''
        for toy_file_name, md5_checksum in file_to_checksum.items():
            path = os.path.join(SCANNED_TOYS_PATH, toy_file_name)
            if not os.path.exists(path):
                if return_reason:
                    msg = f'missing {path}'
                have_all = False
                break
            with open(path, 'rb') as f:
                if md5_checksum != hashlib.md5(f.read()).hexdigest():
                    if return_reason:
                        msg = f'md5 mismatch for {path}, expected: {md5_checksum}'
                    have_all = False
                    break
        if return_reason:
            return have_all, msg
        return have_all

    def _wget_unzip(url):
        file_basename = os.path.basename(url)
        target_filename = os.path.join(SCANNED_TOYS_PATH, file_basename)
        wget.download(url, target_filename)
        with zipfile.ZipFile(target_filename, 'r') as zip_ref:
            zip_ref.extractall(SCANNED_TOYS_PATH)
        os.remove(target_filename)

    def _download_if_needed(url, expected_file_checksums):
        if not _have_expected_files(expected_file_checksums):
            _wget_unzip(url)
            have_all, msg = _have_expected_files(expected_file_checksums, return_reason=True)
            assert have_all, f'After download of {url}, still file mismatch: {msg}'

    if not os.path.exists(SCANNED_TOYS_PATH):
        os.makedirs(SCANNED_TOYS_PATH, exist_ok=True)

    _download_if_needed('https://nvidia-kaolin.s3.us-east-2.amazonaws.com/data/toys_gaussians.usdc.zip',
                        _TOYS_USDC_CHECKSUMS)
    _download_if_needed('https://nvidia-kaolin.s3.us-east-2.amazonaws.com/data/toys_gaussians.pt.zip',
                        _TOYS_PT_CHECKSUMS)
    _download_if_needed('https://nvidia-kaolin.s3.us-east-2.amazonaws.com/data/toys_gaussians.ply.zip',
                        _TOYS_PLY_CHECKSUMS)

    return SCANNED_TOYS_PATH