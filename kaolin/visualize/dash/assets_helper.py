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

"""Helper utilities for serving Kaolin Dash library assets."""

import os
from pathlib import Path
from typing import List

__all__ = [
    'get_kaolin_assets_source_path',
    'get_kaolin_assets_serve_path',
]


def get_kaolin_assets_source_path() -> str:
    """Get the absolute path to Kaolin's library assets folder.
    
    Returns:
        str: Absolute path to the assets directory
    """
    # Get the directory containing this file
    current_dir = Path(__file__).parent
    assets_dir = current_dir / 'distr'

    return str(assets_dir.absolute())

def get_kaolin_assets_serve_path() -> str:
    return '/_kaolin_assets'
