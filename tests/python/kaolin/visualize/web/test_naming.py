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

"""Tests for kaolin.visualize.web.naming."""

import pytest

from kaolin.visualize.web import naming
from kaolin.visualize.web.naming import SequenceWithUniqueGlobalIds, UniqueIdGenerator


@pytest.fixture(autouse=True)
def isolated_generator(monkeypatch):
    monkeypatch.setattr(UniqueIdGenerator, '_instance', UniqueIdGenerator())


class TestUniqueIdGenerator:
    def test_names_prefixes_and_reset(self, monkeypatch):
        generator = UniqueIdGenerator(prefix='view')
        assert generator.get_unique_id_local(' Brush / size ') == 'view-Brush-size'
        assert generator.get_unique_id_local('Brush-size') == 'view-Brush-size_1'
        assert generator.get_unique_id_local('Brush-size') == 'view-Brush-size_2'
        assert generator.get_unique_id_local('Brush-size', prefix='other') == 'other-Brush-size'
        generator.reset_ids(['view-Brush-size'])
        assert generator.get_unique_id_local('Brush-size') == 'view-Brush-size_1'
        generator.reset_ids()
        assert generator.get_unique_id_local('Brush-size') == 'view-Brush-size'
        monkeypatch.setattr(naming, 'generate_random_string', lambda length: 'bcd')
        assert generator.get_unique_id_local() == 'view-bcd'
        assert UniqueIdGenerator.get_unique_id('item') == 'kaolin-item'
        assert UniqueIdGenerator.get_unique_id('item') == 'kaolin-item_1'


class TestSequenceWithUniqueGlobalIds:
    def test_add_lookup_and_update(self):
        sequence = SequenceWithUniqueGlobalIds(uid_prefix='viewer')
        first = sequence.add('paint', {'color': 'red'})
        second = sequence.add(None, 'image', name_hint='background')
        assert (first, second) == ('viewer-paint', 'viewer-background')
        assert sequence[0] == sequence['paint'] == sequence[first] == {'color': 'red'}
        sequence['paint'] = 'updated'
        assert sequence[first] == 'updated'
        assert sequence.get_unique_ids_or_raise([0, 'paint', first]) == [first] * 3
        assert sequence.get_unique_id_or_raise(1) == second
        assert sequence.ids == ['paint']
        assert sequence.unique_ids == [first, second]
        assert sequence.unique_ids_to_items == {first: 0, second: 1}
        snapshot = sequence.items
        snapshot.clear()
        assert sequence.items == ['updated', 'image']
        assert 'paint' in repr(sequence)

    def test_invalid_keys(self):
        sequence = SequenceWithUniqueGlobalIds('viewer')
        uid = sequence.add('paint', 1)
        for identifier in ('paint', uid):
            with pytest.raises(RuntimeError, match='already used'):
                sequence.add(identifier, 2)
        for key, error in [(-1, IndexError), (1, IndexError), ('missing', KeyError), (None, TypeError)]:
            with pytest.raises(error):
                sequence[key]
            with pytest.raises(error):
                sequence[key] = 2
        assert sequence.get_unique_id('missing') is None
        with pytest.raises(ValueError, match='not found'):
            sequence.get_unique_id_or_raise('missing')


def test_generate_random_string(monkeypatch):
    monkeypatch.setattr(naming.secrets, 'choice', lambda alphabet: alphabet[0])
    assert naming.generate_random_string(5) == 'bbbbb'
    assert naming.generate_random_string(0) == ''
