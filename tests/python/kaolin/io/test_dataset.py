# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import pytest
import torch
from torch.utils.data import Dataset

from kaolin.io.dataset import _get_hash, Cache, ProcessedDataset, \
    KaolinDatasetItem, CombinationDataset


def test_cache(tmpdir):
    def func(x):
        return x * 2

    cache_dir = str(tmpdir.join('test_cache'))

    cache = Cache(
        func,
        cache_dir=cache_dir,
        cache_key=_get_hash(repr(func))
    )

    for name, x, target in [
        ('a', torch.tensor([1, 2, 3]), torch.tensor([2, 4, 6])),
        ('b', torch.tensor([0, 5, 8]), torch.tensor([0, 10, 16])),
        ('c', torch.tensor([1, -2, -12]), torch.tensor([2, -4, -24])),
    ]:
        result1 = cache(name, x)
        assert torch.allclose(result1, target)

        result2 = cache(name)
        assert torch.allclose(result2, target)


class TestProcessedDataset(object):
    class TempDataset(Dataset):

        def __init__(self):
            self.data = [
                torch.tensor([1, 2, 3]),
                torch.tensor([0, 5, 8]),
                torch.tensor([1, -2, -12]),
            ]
            self.names = [
                'a', 'b', 'c'
            ]

        def get_data(self, index):
            return self.data[index]

        def get_attributes(self, index):
            return {'name': self.names[index]}

        def get_cache_key(self, index):
            return self.names[index]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return KaolinDatasetItem(self.get_data(index),
                                     self.get_attributes(index))

    class TempPreprocessingTransform:
        def __init__(self):
            self.num_calls = 0

        def __call__(self, x):
            self.num_calls += 1
            return x * 2

    @pytest.fixture(autouse=True)
    def base_dataset_plain(self):
        return [
            torch.tensor([1, 2, 3]),
            torch.tensor([0, 5, 8]),
            torch.tensor([1, -2, -12]),
        ]

    @pytest.fixture(autouse=True)
    def base_dataset_kaolin(self):
        return TestProcessedDataset.TempDataset()

    @pytest.fixture(autouse=True)
    def preprocessing_transform(self):
        return TestProcessedDataset.TempPreprocessingTransform()

    def test_plain_base_dataset(self, base_dataset_plain,
                                preprocessing_transform, tmpdir):
        cache_dir = str(tmpdir.join('test_dataset_cache'))
        d = ProcessedDataset(base_dataset_plain, preprocessing_transform,
                             cache_dir=cache_dir)

        expected = [
            KaolinDatasetItem(data=torch.tensor([2, 4, 6]), attributes={}),
            KaolinDatasetItem(data=torch.tensor([0, 10, 16]), attributes={}),
            KaolinDatasetItem(data=torch.tensor([2, -4, -24]), attributes={}),
        ]

        before = preprocessing_transform.num_calls

        for i in range(len(d)):
            a = d[i]
            b = expected[i]
            assert torch.allclose(a.data, b.data)
            assert a.attributes == b.attributes

        assert preprocessing_transform.num_calls == before

    def test_kaolin_base_dataset(self, base_dataset_kaolin,
                                 preprocessing_transform, tmpdir):
        cache_dir = str(tmpdir.join('test_dataset_cache'))
        d = ProcessedDataset(base_dataset_kaolin, preprocessing_transform,
                             cache_dir=cache_dir)

        expected = [
            KaolinDatasetItem(data=torch.tensor([2, 4, 6]),
                              attributes={'name': 'a'}),
            KaolinDatasetItem(data=torch.tensor([0, 10, 16]),
                              attributes={'name': 'b'}),
            KaolinDatasetItem(data=torch.tensor([2, -4, -24]),
                              attributes={'name': 'c'}),
        ]

        before = preprocessing_transform.num_calls

        for i in range(len(d)):
            a = d[i]
            b = expected[i]
            assert torch.allclose(a.data, b.data)
            assert a.attributes == b.attributes

        assert preprocessing_transform.num_calls == before


class TestCombinationDataset(object):

    @pytest.fixture(autouse=True)
    def base_dataset_kaolin(self):
        class TempDataset(Dataset):

            def __init__(self):
                self.data = [
                    torch.tensor([1]),
                    torch.tensor([2]),
                    torch.tensor([3]),
                ]
                self.names = [
                    'a', 'b', 'c'
                ]

            def get_data(self, index):
                return self.data[index]

            def get_attributes(self, index):
                return {'name': self.names[index]}

            def get_cache_key(self, index):
                return self.names[index]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                return KaolinDatasetItem(self.get_data(index),
                                         self.get_attributes(index))

        return TempDataset()

    @pytest.fixture(autouse=True)
    def base_dataset_plain(self):
        return [
            torch.tensor([4]),
            torch.tensor([5]),
            torch.tensor([6]),
        ]

    def test_combination_dataset(self, base_dataset_kaolin, base_dataset_plain):
        d = CombinationDataset((base_dataset_kaolin, base_dataset_plain))

        expected = [
            KaolinDatasetItem(data=(torch.tensor([1]), torch.tensor([4])),
                              attributes=({'name': 'a'}, {})),
            KaolinDatasetItem(data=(torch.tensor([2]), torch.tensor([5])),
                              attributes=({'name': 'b'}, {})),
            KaolinDatasetItem(data=(torch.tensor([3]), torch.tensor([6])),
                              attributes=({'name': 'c'}, {})),
        ]

        for i in range(len(d)):
            a = d[i]
            b = expected[i]
            assert len(a.data) == len(b.data)
            assert torch.allclose(a.data[0], b.data[0])
            assert torch.allclose(a.data[1], b.data[1])
            assert a.attributes == b.attributes
