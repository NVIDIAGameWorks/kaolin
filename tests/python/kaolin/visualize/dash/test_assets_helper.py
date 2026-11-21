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

"""Tests for kaolin.visualize.dash.assets_helper."""

from pathlib import Path

from kaolin.visualize.dash import assets_helper


def test_assets_source_path():
    path = Path(assets_helper.get_kaolin_assets_source_path())
    assert path.is_absolute()
    assert path == Path(assets_helper.__file__).parent / 'distr'
    assert path.is_dir()
    assert (path / 'favicon.ico').is_file()


def test_assets_serve_path():
    assert assets_helper.get_kaolin_assets_serve_path() == '/_kaolin_assets'
