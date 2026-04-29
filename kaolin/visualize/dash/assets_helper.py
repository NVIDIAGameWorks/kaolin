# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.

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
