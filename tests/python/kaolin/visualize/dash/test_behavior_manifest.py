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

"""Tests for kaolin.visualize.dash.behavior_manifest."""

import json

import pytest

from kaolin.visualize.dash import behavior_manifest as manifest
from kaolin.visualize.dash.option import BehaviorMeta, OptionKind


@pytest.fixture
def resource(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, '_loaded', False)
    monkeypatch.setattr(manifest, '_cache', {})
    monkeypatch.setattr(manifest, '_meta_cache', None)
    monkeypatch.setattr(manifest.resources, 'files', lambda package: tmp_path)
    return tmp_path / manifest.BEHAVIOR_MANIFEST_RESOURCE


def test_manifest_lookup_and_cache(resource):
    resource.write_text(json.dumps({'brush': {
        'description': 'Paint', 'options': {'size': {'kind': 'int', 'default': 3}}}}))
    entries = manifest.get_behavior_manifest()
    meta = manifest.get_behavior_meta('brush')
    assert isinstance(meta, BehaviorMeta) and entries['brush'] is meta
    assert meta.description == 'Paint'
    assert (meta.options['size'].kind, meta.options['size'].default) == (OptionKind.INT, 3)
    assert manifest.get_behavior_meta('missing') is None
    resource.write_text('{}')
    assert manifest.get_behavior_manifest() is entries


@pytest.mark.parametrize('contents,message', [
    (None, 'not found'), ('{', 'not valid JSON'), ('[]', 'must be an object')])
def test_unavailable_manifest(resource, caplog, contents, message):
    if contents is not None:
        resource.write_text(contents)
    assert manifest.get_behavior_manifest() == {}
    assert message in caplog.text
    caplog.clear()
    assert manifest.get_behavior_manifest() == {}
    assert not caplog.records


def test_invalid_entry_is_skipped(resource, caplog):
    resource.write_text(json.dumps({'good': {'options': {}}, 'bad': {'unknown': True}}))
    assert list(manifest.get_behavior_manifest()) == ['good']
    assert 'cannot decode entry' in caplog.text
