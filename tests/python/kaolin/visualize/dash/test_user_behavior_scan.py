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

"""Tests for kaolin.visualize.dash.user_behavior_scan."""

import os
from pathlib import Path

import pytest

from kaolin.visualize.dash import user_behavior_scan as scan
from kaolin.visualize.dash.option import BehaviorMeta


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(scan, '_file_cache', {})


def test_scan_user_behavior_names(tmp_path):
    nested = tmp_path / 'nested'
    nested.mkdir()
    for suffix in ('js', 'ts', 'tsx'):
        (nested / f'custom.{suffix}').write_text(
            f'window.kaolin.BehaviorRegister.register("custom_{suffix}", Custom);\n'
            '// BehaviorRegister.register("comment", Custom);\n'
            '/* BehaviorRegister.register("example", Custom); */')
    (tmp_path / 'ignored.py').write_text('BehaviorRegister.register("ignored", Custom);')
    entries = scan.scan_user_behavior_names([tmp_path])
    assert set(entries) == {'custom_js', 'custom_ts', 'custom_tsx'}
    for suffix in ('js', 'ts', 'tsx'):
        meta = entries[f'custom_{suffix}']
        assert isinstance(meta, BehaviorMeta)
        assert meta.source_path == str(nested / f'custom.{suffix}')
        assert meta.description is None and meta.options == {}


def test_missing_directory(tmp_path, caplog):
    assert scan.scan_user_behavior_names([tmp_path / 'missing']) == {}
    assert 'does not exist' in caplog.text


def test_duplicate_names(tmp_path, caplog):
    directories = [tmp_path / 'first', tmp_path / 'second']
    for directory in directories:
        directory.mkdir()
        (directory / 'custom.js').write_text('BehaviorRegister.register("custom", Custom);')
    entries = scan.scan_user_behavior_names(directories)
    assert entries['custom'].source_path == str(directories[1] / 'custom.js')
    assert 'registered in both' in caplog.text


def test_file_cache_refreshes_on_modification(tmp_path):
    asset = tmp_path / 'custom.js'
    asset.write_text('BehaviorRegister.register("first", Custom);')
    assert set(scan.scan_user_behavior_names([tmp_path])) == {'first'}
    assert set(scan.scan_user_behavior_names([tmp_path])) == {'first'}
    mtime = asset.stat().st_mtime
    asset.write_text('BehaviorRegister.register("second", Custom);')
    os.utime(asset, (mtime + 2, mtime + 2))
    assert set(scan.scan_user_behavior_names([tmp_path])) == {'second'}


def test_unreadable_asset(tmp_path, monkeypatch, caplog):
    asset = tmp_path / 'unreadable.js'
    asset.write_text('BehaviorRegister.register("custom", Custom);')
    read_text = Path.read_text

    def read(path, *args, **kwargs):
        if path == asset:
            raise PermissionError('unreadable asset')
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)
    assert scan.scan_user_behavior_names([tmp_path]) == {}
    assert 'cannot read' in caplog.text and str(asset) in caplog.text


def test_asset_disappears_during_scan(tmp_path, monkeypatch, caplog):
    asset = tmp_path / 'removed.js'
    asset.write_text('BehaviorRegister.register("custom", Custom);')
    is_file = Path.is_file

    def remove_after_check(path):
        result = is_file(path)
        if path == asset:
            path.unlink()
        return result

    monkeypatch.setattr(Path, 'is_file', remove_after_check)
    assert scan.scan_user_behavior_names([tmp_path]) == {}
    assert 'cannot stat' in caplog.text and str(asset) in caplog.text
