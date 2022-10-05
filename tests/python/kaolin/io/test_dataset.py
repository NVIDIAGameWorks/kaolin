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

import os
import glob
from pathlib import Path
import time
import shutil

import pytest
import torch
from torch.utils.data import Dataset

from kaolin.utils.testing import contained_torch_equal
from kaolin.io.dataset import CachedDataset
from kaolin.io.dataset import ProcessedDataset, Cache, _get_hash, \
    _preprocess_task, CombinationDataset, KaolinDatasetItem

class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.threshold = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.threshold is not None and idx >= self.threshold:
            raise RuntimeError("this is an error.")
        return self.data[idx]


class DummyTransform(object):
    def __init__(self, offset, add_sum):
        self.offset = offset
        self.add_sum = add_sum

    def __call__(self, inputs):
        outputs = {'c': 0} if self.add_sum else {}
        for k, v in inputs.items():
            outputs[k] = v + self.offset
            if self.add_sum:
                outputs['c'] += v
        return outputs

CACHE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_out_dataset')

# TODO(cfujitsang): We are technically not testing that the ProcessedDataset is
# really using the cache files, not sure how to do that tho
class TestCachedDataset:
    @pytest.fixture(autouse=True, scope='class')
    def dataset_size(self):
        return 10

    @pytest.fixture(autouse=True)
    def dummy_dataset(self, dataset_size):
        return DummyDataset([{'a': 1 + i * 2, 'b': (i + 4) * 3} for i in range(dataset_size)])

    @pytest.fixture(autouse=True)
    def larger_dataset(self, dataset_size):
        return DummyDataset([{'a': 1 + i * 2, 'b': (i + 4) * 3} for i in range(dataset_size + 2)])

    @pytest.fixture(autouse=True)
    def diff_val_dataset(self, dataset_size):
        return DummyDataset([{'a': 2 + i * 2, 'b': (i + 4) * 3 + 1} for i in range(dataset_size)])

    @pytest.fixture(autouse=True)
    def diff_fields_dataset(self, dataset_size):
        return DummyDataset([{'a': 1 + i * 2, 'c': (i + 10) * 2} for i in range(dataset_size)])

    def setup_method(self, method):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    def teardown_method(self, method):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    @pytest.mark.parametrize('use_transform', [True, False])
    @pytest.mark.parametrize('cache_at_runtime', [True, False])
    @pytest.mark.parametrize('num_workers', [0, 4])
    @pytest.mark.parametrize('force_overwrite', [True, False])
    def test_no_save_on_disk(self, dataset_size, dummy_dataset, use_transform,
                             cache_at_runtime, num_workers, force_overwrite):
        assert not os.path.exists(CACHE_DIR)
        transform = DummyTransform(10, True) if use_transform is None else None
        ds = CachedDataset(dummy_dataset,
                           transform=transform,
                           cache_dir=None,
                           save_on_disk=False,
                           num_workers=num_workers,
                           force_overwrite=force_overwrite,
                           cache_at_runtime=cache_at_runtime)
        # Check outputs of the processed dataset
        for i in range(dataset_size):
            assert ds[i] == dummy_dataset[i]
        assert not os.path.exists(CACHE_DIR)
 
    @pytest.mark.parametrize('use_transform', [True, False])
    @pytest.mark.parametrize('cache_at_runtime', [True, False])
    @pytest.mark.parametrize('num_workers', [0, 4])
    @pytest.mark.parametrize('force_overwrite', [True, False])
    def test_all_save_on_disk(self, dataset_size, dummy_dataset, use_transform,
                              cache_at_runtime, num_workers, force_overwrite):
        assert not os.path.exists(CACHE_DIR)
        transform = DummyTransform(10, True) if use_transform else None
        ds = CachedDataset(dummy_dataset,
                           transform=transform,
                           cache_dir=CACHE_DIR,
                           save_on_disk=True,
                           num_workers=num_workers,
                           force_overwrite=force_overwrite,
                           cache_at_runtime=cache_at_runtime)
        assert os.path.exists(CACHE_DIR)
        all_expected_files = set()
        # Check outputs of dataset
        for i in range(dataset_size):
            expected_data = dummy_dataset[i]
            fields = list(expected_data.keys())
            expected_files = list(os.path.join(CACHE_DIR, str(i), f'{k}.pt') for k in fields)
            all_expected_files.update(expected_files)
            if cache_at_runtime:
                assert all(not os.path.exists(path) for path in expected_files)
            expected_cache = dummy_dataset[i]
            expected_output = transform(expected_cache) if use_transform else expected_cache
            output = ds[i]
            # Check that the cached files correspond to post-preprocessing outputs
            assert expected_cache == {
                k: torch.load(path) for path, k in zip(expected_files, fields)
            }
            assert contained_torch_equal(output, expected_output)

        assert set(glob.glob(os.path.join(CACHE_DIR, '*', '*'))) == all_expected_files

        # Check if dataset is reusing the cached files (or force overwrite)
        files_mtime = {path: os.path.getmtime(path) for path in all_expected_files}
        time.sleep(0.01) # This is just in case I/O is so fast that files get greated in the same time
        ds2 = CachedDataset(dummy_dataset,
                            transform=transform,
                            cache_dir=CACHE_DIR,
                            save_on_disk=True,
                            num_workers=num_workers,
                            force_overwrite=force_overwrite,
                            cache_at_runtime=cache_at_runtime)
        if cache_at_runtime:
            for path, mtime in files_mtime.items():
                assert os.path.getmtime(path) == mtime
            for _ in ds2:
                pass

        for path, mtime in files_mtime.items():
            if force_overwrite:
                assert os.path.getmtime(path) != mtime
            else:
                assert os.path.getmtime(path) == mtime

    @pytest.mark.parametrize('use_transform', [True, False])
    @pytest.mark.parametrize('cache_at_runtime', [True, False])
    @pytest.mark.parametrize('num_workers', [0, 4])
    @pytest.mark.parametrize('save_on_disk', [['b'], ['a', 'b']])
    @pytest.mark.parametrize('force_overwrite', [True, False])
    def test_set_save_on_disk(self, dataset_size, dummy_dataset, use_transform,
                              num_workers, save_on_disk, force_overwrite, cache_at_runtime):
        assert not os.path.exists(CACHE_DIR)
        transform = DummyTransform(10, True) if use_transform else None
        ds = CachedDataset(dummy_dataset,
                           transform=transform,
                           cache_dir=CACHE_DIR,
                           save_on_disk=save_on_disk,
                           num_workers=num_workers,
                           force_overwrite=force_overwrite,
                           cache_at_runtime=cache_at_runtime)
        all_expected_files = set(os.path.join(CACHE_DIR, str(i), f'{k}.pt') for k in save_on_disk
                                 for i in range(dataset_size))
        # Check output of dataset
        for i in range(dataset_size):
            expected_cache = dummy_dataset[i]
            expected_files = list(os.path.join(CACHE_DIR, str(i), f'{k}.pt') for k in save_on_disk)
            if cache_at_runtime:
                assert all(not os.path.exists(path) for path in expected_files)
            expected_output = transform(expected_cache) if use_transform else expected_cache
            output = ds[i]
            # Check that the cached files correspond to post-preprocessing outputs
            for k in save_on_disk:
                assert expected_cache[k] == torch.load(os.path.join(CACHE_DIR, str(i), f'{k}.pt'))
            assert output == expected_output

        assert set(glob.glob(os.path.join(CACHE_DIR, '*', '*'))) == all_expected_files

        # Check that same dataset will reuse the cached files (won't overwrite)
        files_mtime = {path: os.path.getmtime(path) for path in all_expected_files}
        time.sleep(0.01) # This is just in case I/O is so fast that files get greated in the same time
        ds2 = CachedDataset(dummy_dataset,
                            transform=transform,
                            cache_dir=CACHE_DIR,
                            save_on_disk=save_on_disk,
                            num_workers=num_workers,
                            force_overwrite=force_overwrite,
                            cache_at_runtime=cache_at_runtime)
        if cache_at_runtime:
            for path, mtime in files_mtime.items():
                assert os.path.getmtime(path) == mtime
            for _ in ds2:
                pass
        for path, mtime in files_mtime.items():
            if force_overwrite:
                assert os.path.getmtime(path) != mtime
            else:
                assert os.path.getmtime(path) == mtime

    @pytest.mark.parametrize('use_transform', [True, False])
    @pytest.mark.parametrize('cache_at_runtime', [True, False])
    @pytest.mark.parametrize('num_workers', [0, 4])
    @pytest.mark.parametrize('save_on_disk', [True, ['b']])
    def test_fail_larger_dataset(self, dataset_size, dummy_dataset, larger_dataset,
                                 use_transform, num_workers, save_on_disk, cache_at_runtime):
        assert not os.path.exists(CACHE_DIR)

        transform = DummyTransform(10, True) if use_transform else None
        ds = CachedDataset(dummy_dataset,
                           transform=transform,
                           cache_dir=CACHE_DIR,
                           save_on_disk=save_on_disk,
                           num_workers=num_workers,
                           cache_at_runtime=cache_at_runtime)
        if cache_at_runtime:
            for _ in ds:
                pass
        
        expected_files = glob.glob(os.path.join(CACHE_DIR, '*', '*.pt'))
        files_mtime = {path: os.path.getmtime(path) for path in expected_files}
        time.sleep(0.01) # This is just in case I/O is so fast that files get greated in the same time

        # An error should be raised in the size of the dataset is different than the number of cached subfolder
        # using a POSIX path to avoid regex issues with Windows path backslashes.
        #   this is only a problem for the `pytest.raises` regex logic, not a problem with file saving.
        cache_check_str = Path(CACHE_DIR).resolve().as_posix()
        with pytest.raises(RuntimeError,
                        #    match=f"{len(dummy_dataset)} files already exist "):
                           match=f"{len(dummy_dataset)} files already exist on {cache_check_str} "
                                 f"this dataset as {len(larger_dataset)} files so caching "
                                 "is too ambiguous and error-prone please force rewriting "
                                 "by setting 'force_overwrite'"):
            larger_ds = CachedDataset(larger_dataset,
                                      transform=transform,
                                      cache_dir=CACHE_DIR,
                                      save_on_disk=save_on_disk,
                                      num_workers=num_workers,
                                      cache_at_runtime=cache_at_runtime)
        # Nothing got overwriten before throwing the error
        for path, mtime in files_mtime.items():
            assert os.path.getmtime(path) == mtime

        files_mtime = {path: os.path.getmtime(path) for path in expected_files}
        ds2 = CachedDataset(larger_dataset,
                            transform=transform,
                            cache_dir=CACHE_DIR,
                            save_on_disk=save_on_disk,
                            num_workers=num_workers,
                            force_overwrite=True,
                            cache_at_runtime=cache_at_runtime)
        if cache_at_runtime:
            assert len(glob.glob(os.path.join(CACHE_DIR, '*'))) == 0
            for _ in ds2:
                pass
        
        # with 'force_overwrite' the cache files should be overwritten
        for path, mtime in files_mtime.items():
            assert os.path.getmtime(path) != mtime

    @pytest.mark.parametrize('use_transform', [True, False])
    @pytest.mark.parametrize('cache_at_runtime', [True, False])
    @pytest.mark.parametrize('num_workers', [0, 4])
    @pytest.mark.parametrize('save_on_disk', [True, ['b']])
    def test_fail_diff_val_dataset(self, dummy_dataset, diff_val_dataset, use_transform,
                                   num_workers, save_on_disk, cache_at_runtime):
        assert not os.path.exists(CACHE_DIR)

        

        transform = DummyTransform(10, True) if use_transform else None
        ds = CachedDataset(dummy_dataset,
                           transform=transform,
                           cache_dir=CACHE_DIR,
                           save_on_disk=save_on_disk,
                           num_workers=num_workers,
                           cache_at_runtime=cache_at_runtime)

        if cache_at_runtime:
            for _ in ds:
                pass
        expected_files = glob.glob(os.path.join(CACHE_DIR, '*', '*.pt'))
        files_mtime = {path: os.path.getmtime(path) for path in expected_files}
        time.sleep(0.01) # This is just in case I/O is so fast that files get greated in the same time
        f"file '{os.path.join(CACHE_DIR, '0')}"
        
        # using a POSIX path to avoid regex issues with Windows path backslashes.
        #   this is only a problem for the `pytest.raises` regex logic, not a problem with file saving.
        cache_check_str = (Path(CACHE_DIR) / "0").resolve().as_posix()
        with pytest.raises(AssertionError,
                           match=f"file '{cache_check_str}"
                                 r"/.\.pt' is different than its matching field from the "
                                 "input dataset, set 'force_overwriting' to True "
                                 "to overwrite the files cached."):
            diff_ds = CachedDataset(diff_val_dataset,
                                    transform=transform,
                                    cache_dir=CACHE_DIR,
                                    save_on_disk=save_on_disk,
                                    num_workers=num_workers,
                                    cache_at_runtime=cache_at_runtime)

        # Nothing got overwriten before throwing the error
        for path, mtime in files_mtime.items():
            assert os.path.getmtime(path) == mtime

        # Check that same dataset will reuse the cached files (won't overwrite)
        files_mtime = {path: os.path.getmtime(path) for path in expected_files}
        ds2 = CachedDataset(diff_val_dataset,
                            transform=transform,
                            cache_dir=CACHE_DIR,
                            save_on_disk=save_on_disk,
                            num_workers=num_workers,
                            force_overwrite=True,
                            cache_at_runtime=cache_at_runtime)

        if cache_at_runtime:
            for path, mtime in files_mtime.items():
                assert os.path.getmtime(path) == mtime
            for _ in ds2:
                pass
 
        # with 'force_overwrite' the cache files should be overwritten
        for path, mtime in files_mtime.items():
            assert os.path.getmtime(path) != mtime

    @pytest.mark.parametrize('use_transform', [True, False])
    @pytest.mark.parametrize('cache_at_runtime', [True, False])
    @pytest.mark.parametrize('num_workers', [0, 4])
    @pytest.mark.parametrize('save_on_disk', [True, ['a', 'c']])
    @pytest.mark.parametrize('force_overwrite', [True, False])
    def test_fail_diff_fields_dataset(self, dummy_dataset, diff_fields_dataset, num_workers,
                                      use_transform, cache_at_runtime, save_on_disk, force_overwrite):
        assert not os.path.exists(CACHE_DIR)

        transform = DummyTransform(10, True) if use_transform else None
        ds = CachedDataset(dummy_dataset,
                           transform=transform,
                           cache_dir=CACHE_DIR,
                           save_on_disk=True,
                           num_workers=num_workers,
                           cache_at_runtime=cache_at_runtime)

        if cache_at_runtime:
            for _ in ds:
                pass
        orig_files = glob.glob(os.path.join(CACHE_DIR, '*', '*.pt'))
        files_mtime = {path: os.path.getmtime(path) for path in orig_files}
        time.sleep(0.01) # This is just in case I/O is so fast that files get greated in the same time
        diff_ds = CachedDataset(diff_fields_dataset,
                                transform=transform,
                                cache_dir=CACHE_DIR,
                                save_on_disk=save_on_disk,
                                num_workers=num_workers,
                                force_overwrite=force_overwrite,
                                cache_at_runtime=cache_at_runtime)

        if cache_at_runtime:
            for _ in diff_ds:
                pass

        for path in glob.glob(os.path.join(CACHE_DIR, '*', 'a.pt')):
            # a.pt should only be overwritten if 'force_overwrite' is True
            if force_overwrite:
                assert os.path.getmtime(path) != files_mtime[path]
            else:
                assert os.path.getmtime(path) == files_mtime[path]

        for path in glob.glob(os.path.join(CACHE_DIR, '*', 'b.pt')):
            assert os.path.getmtime(path) == files_mtime[path]

    @pytest.mark.parametrize('use_transform', [True, False])
    @pytest.mark.parametrize('cache_at_runtime', [True, False])
    @pytest.mark.parametrize('num_workers', [0, 4])
    @pytest.mark.parametrize('save_on_disk', [True, ['a', 'c']])
    @pytest.mark.parametrize('force_overwrite', [True, False])
    def test_clean_on_fail(self, dummy_dataset, diff_fields_dataset, dataset_size, use_transform,
                           num_workers, cache_at_runtime, save_on_disk, force_overwrite):
        assert not os.path.exists(CACHE_DIR)

        transform = DummyTransform(10, True) if use_transform else None
        ds = CachedDataset(dummy_dataset,
                           transform=transform,
                           cache_dir=CACHE_DIR,
                           save_on_disk=True,
                           num_workers=num_workers,
                           cache_at_runtime=cache_at_runtime)

        if cache_at_runtime:
            for _ in ds:
                pass
        orig_files = glob.glob(os.path.join(CACHE_DIR, '*', '*.pt'))
        files_mtime = {path: os.path.getmtime(path) for path in orig_files}
        time.sleep(0.01) # This is just in case I/O is so fast that files get greated in the same time
        # to raise an error on purpose
        diff_fields_dataset.threshold = 5
        #with pytest.raises(RuntimeError, match="this is an error."):
        try:
            diff_ds = CachedDataset(diff_fields_dataset,
                                    transform=transform,
                                    cache_dir=CACHE_DIR,
                                    save_on_disk=save_on_disk,
                                    num_workers=num_workers,
                                    force_overwrite=force_overwrite,
                                    cache_at_runtime=cache_at_runtime)
        except Exception:
            pass

        if cache_at_runtime:
            for i in range(5):
                _ = diff_ds[i]
            assert len(glob.glob(os.path.join(CACHE_DIR, '*', 'c.pt'))) == 5
            try:
                _ = diff_ds[5]
            except Exception as e:
                pass

        assert len(glob.glob(os.path.join(CACHE_DIR, '*', 'c.pt'))) == 0

        if force_overwrite:
            assert len(glob.glob(os.path.join(CACHE_DIR, '*', 'a.pt'))) == 0
        else:
            assert len(glob.glob(os.path.join(CACHE_DIR, '*', 'a.pt'))) == dataset_size

        assert len(glob.glob(os.path.join(CACHE_DIR, '*', 'b.pt'))) == dataset_size


    @pytest.mark.parametrize('use_transform', [True, False])
    @pytest.mark.parametrize('cache_at_runtime', [True, False])
    @pytest.mark.parametrize('num_workers', [0, 4])
    @pytest.mark.parametrize('save_on_disk', [True, ['a', 'b']])
    @pytest.mark.parametrize('force_overwrite', [True, False])
    def test_full_clean_on_fail(self, dummy_dataset, dataset_size, num_workers, use_transform,
                                cache_at_runtime, save_on_disk, force_overwrite):
        assert not os.path.exists(CACHE_DIR)
        transform = DummyTransform(10, True) if use_transform else None
        if force_overwrite:
            ds = CachedDataset(dummy_dataset,
                               transform=transform,
                               cache_dir=CACHE_DIR,
                               save_on_disk=True,
                               num_workers=num_workers,
                               cache_at_runtime=cache_at_runtime)

            if cache_at_runtime:
                for _ in ds:
                    pass

        time.sleep(0.01) # This is just in case I/O is so fast that files get greated in the same time
        # to raise an error on purpose
        dummy_dataset.threshold = 5
        #with pytest.raises(RuntimeError, match="this is an error."):
        try:
            ds2 = CachedDataset(dummy_dataset,
                                transform=transform,
                                cache_dir=CACHE_DIR,
                                save_on_disk=save_on_disk,
                                num_workers=num_workers,
                                force_overwrite=force_overwrite,
                                cache_at_runtime=cache_at_runtime)
        except Exception:
            pass

        if cache_at_runtime:
            for i in range(5):
                _ = ds2[i]
            try:
                _ = ds2[5]
            except Exception as e:
                pass

        assert not os.path.exists(CACHE_DIR)

#class TestProcessedDataset:
#    @pytest.fixture(autouse=True, scope='class')
#    def dummy_dataset(self):
#        return DummyDataset([{'a': 1 + i * 2, 'b': (i + 4) * 3} for i in range(10)])
#
#    @pytest.fixture(autouse=True, scope='class')
#    def diff_dataset(self):
#        return DummyDataset([{'a': 2 + i * 2, 'b': (i + 4) * 3 + 1} for i in range(10)])
#
#    @pytest.fixture(autouse=True, scope='class')
#    def larger_dataset(self):
#        return DummyDataset([{'a': 1 + i * 2, 'b': (i + 4) * 3} for i in range(12)])
#
#    def setup_method(self, method):
#        print("SETUP")
#        shutil.rmtree(CACHE_DIR, ignore_errors=True)
#
#    def teardown_method(self, method):
#        print("TEARDOWN")
#        shutil.rmtree(CACHE_DIR, ignore_errors=True)
#
#    @pytest.mark.parametrize('preprocess_at_runtime', [True, False])
#    @pytest.mark.parametrize('force_overwrite', [True, False])
#    @pytest.mark.parametrize('use_preprocessing', [True, False])
#    @pytest.mark.parametrize('use_transform', [True, False])
#    @pytest.mark.parametrize('num_workers', [0, 4])
#    def test_no_save_on_disk(self, dummy_dataset, num_workers, use_preprocessing, 
#                             use_transform, force_overwrite, preprocess_at_runtime):
#        assert not os.path.exists(CACHE_DIR)
#        transform = DummyTransform(10., False) if use_transform else None
#        preprocessing_transform = DummyTransform(100., True) if use_preprocessing else None
#        ds = ProcessedDatasetV2(dummy_dataset, cache_dir=None, save_on_disk=False,
#                                preprocessing_transform=preprocessing_transform,
#                                transform=transform, num_workers=num_workers,
#                                force_overwrite=force_overwrite,
#                                preprocess_at_runtime=preprocess_at_runtime)
#        # Check outputs of the processed dataset
#        for i in range(10):
#            expected_data = dummy_dataset[i]
#            if use_preprocessing:
#                expected_data = preprocessing_transform(expected_data)
#            if use_transform:
#                expected_data = transform(expected_data)
#            assert ds[i] == expected_data
#        assert not os.path.exists(CACHE_DIR)
#
#    @pytest.mark.parametrize('preprocess_at_runtime', [True, False])
#    @pytest.mark.parametrize('use_preprocessing', [True, False])
#    @pytest.mark.parametrize('use_transform', [True, False])
#    @pytest.mark.parametrize('num_workers', [0, 4])
#    @pytest.mark.parametrize('force_overwrite', [True, False])
#    def test_all_save_on_disk(self, dummy_dataset, num_workers, use_preprocessing,
#                              use_transform, force_overwrite, preprocess_at_runtime):
#        assert not os.path.exists(CACHE_DIR)
#        transform = DummyTransform(10., False) if use_transform else None
#        preprocessing_transform = DummyTransform(100., True) if use_preprocessing else None
#        ds = ProcessedDatasetV2(dummy_dataset, cache_dir=CACHE_DIR, save_on_disk=True,
#                                transform=transform, num_workers=num_workers,
#                                preprocessing_transform=preprocessing_transform,
#                                force_overwrite=force_overwrite,
#                                preprocess_at_runtime=preprocess_at_runtime)
#        assert os.path.exists(CACHE_DIR)
#        all_expected_files = set()
#        # Check outputs of dataset
#        for i in range(10):
#            expected_data = dummy_dataset[i]
#            if use_preprocessing:
#                expected_data = preprocessing_transform(expected_data)
#            fields = list(expected_data.keys())
#            expected_files = list(os.path.join(CACHE_DIR, str(i), f'{k}.pt') for k in fields)
#            all_expected_files.update(expected_files)
#            if preprocess_at_runtime:
#                assert all(not os.path.exists(path) for path in expected_files)
#            output_data = ds[i]
#            # Check that the cached files correspond to post-preprocessing outputs
#            assert expected_data == {
#                k: torch.load(path) for path, k in zip(expected_files, fields)
#            }
#            if use_transform:
#                expected_data = transform(expected_data)
#            assert output_data == expected_data
#
#        assert set(glob.glob(os.path.join(CACHE_DIR, '*', '*'))) == all_expected_files
#
#        # Check if dataset is reusing the cached files (or force overwrite)
#        files_mtime = {path: os.path.getmtime(path) for path in all_expected_files}
#        time.sleep(0.01) # This is just in case I/O is so fast that files get greated in the same time
#        ds2 = ProcessedDatasetV2(dummy_dataset, cache_dir=CACHE_DIR, save_on_disk=True,
#                                 transform=transform, num_workers=num_workers,
#                                 preprocessing_transform=preprocessing_transform,
#                                 force_overwrite=force_overwrite,
#                                 preprocess_at_runtime=preprocess_at_runtime)
#        if preprocess_at_runtime:
#            for path, mtime in files_mtime.items():
#                assert os.path.getmtime(path) == mtime
#            for _ in ds2:
#                pass
#
#        for path, mtime in files_mtime.items():
#            if force_overwrite:
#                assert os.path.getmtime(path) != mtime
#            else:
#                assert os.path.getmtime(path) == mtime
#
#    @pytest.mark.parametrize('preprocess_at_runtime', [True, False])
#    @pytest.mark.parametrize('use_preprocessing', [True, False])
#    @pytest.mark.parametrize('use_transform', [True, False])
#    @pytest.mark.parametrize('num_workers', [0, 4])
#    @pytest.mark.parametrize('save_on_disk', [['b'], ['a', 'b'], ['a', 'c']])
#    @pytest.mark.parametrize('force_overwrite', [True, False])
#    def test_set_save_on_disk(self, dummy_dataset, num_workers, save_on_disk,
#                              use_preprocessing, use_transform, force_overwrite,
#                              preprocess_at_runtime):
#        if not use_preprocessing and 'c' in save_on_disk:
#            pytest.skip("skipping because field 'c' doesn't exist without preprocessing")
#        assert not os.path.exists(CACHE_DIR)
#
#        transform = DummyTransform(10., False) if use_transform else None
#        preprocessing_transform = DummyTransform(100., True) if use_preprocessing else None
#        ds = ProcessedDatasetV2(dummy_dataset, cache_dir=CACHE_DIR, save_on_disk=save_on_disk,
#                                num_workers=num_workers, transform=transform,
#                                preprocessing_transform=preprocessing_transform,
#                                force_overwrite=force_overwrite,
#                                preprocess_at_runtime=preprocess_at_runtime)
#        all_expected_files = set(os.path.join(CACHE_DIR, str(i), f'{k}.pt') for k in save_on_disk
#                                 for i in range(10))
#        # Check output of dataset
#        for i in range(10):
#            expected_data = dummy_dataset[i]
#            if use_preprocessing:
#                expected_data = preprocessing_transform(expected_data)
#            expected_files = list(os.path.join(CACHE_DIR, str(i), f'{k}.pt') for k in save_on_disk)
#            if preprocess_at_runtime:
#                assert all(not os.path.exists(path) for path in expected_files)
#            output_data = ds[i]
#            # Check that the cached files correspond to post-preprocessing outputs
#            for k in save_on_disk:
#                assert expected_data[k] == torch.load(os.path.join(CACHE_DIR, str(i), f'{k}.pt'))
#            if use_transform:
#                expected_data = transform(expected_data)
#            assert output_data == expected_data
#
#        assert set(glob.glob(os.path.join(CACHE_DIR, '*', '*'))) == all_expected_files
#
#        # Check that same dataset will reuse the cached files (won't overwrite)
#        files_mtime = {path: os.path.getmtime(path) for path in all_expected_files}
#        time.sleep(0.01) # This is just in case I/O is so fast that files get greated in the same time
#        ds2 = ProcessedDatasetV2(dummy_dataset, cache_dir=CACHE_DIR, save_on_disk=save_on_disk,
#                                 transform=transform, num_workers=num_workers,
#                                 preprocessing_transform=preprocessing_transform,
#                                 force_overwrite=force_overwrite,
#                                 preprocess_at_runtime=preprocess_at_runtime)
#        if preprocess_at_runtime:
#            for path, mtime in files_mtime.items():
#                assert os.path.getmtime(path) == mtime
#            for _ in ds2:
#                pass
#        for path, mtime in files_mtime.items():
#            if force_overwrite:
#                assert os.path.getmtime(path) != mtime
#            else:
#                assert os.path.getmtime(path) == mtime
#
#    @pytest.mark.parametrize('preprocess_at_runtime', [True, False])
#    @pytest.mark.parametrize('use_preprocessing', [True, False])
#    @pytest.mark.parametrize('use_transform', [True, False])
#    @pytest.mark.parametrize('num_workers', [0, 4])
#    @pytest.mark.parametrize('save_on_disk', [True, ['b'], ['a', 'c']])
#    def test_fail_larger_dataset(self, dummy_dataset, larger_dataset, num_workers, save_on_disk,
#                                 use_preprocessing, use_transform, preprocess_at_runtime):
#        if not use_preprocessing and isinstance(save_on_disk, list) and 'c' in save_on_disk:
#            pytest.skip("skipping because field 'c' doesn't exist without preprocessing")
#        assert not os.path.exists(CACHE_DIR)
#
#        transform = DummyTransform(10., False) if use_transform else None
#        preprocessing_transform = DummyTransform(100., True) if use_preprocessing else None
#        ds = ProcessedDatasetV2(dummy_dataset, cache_dir=CACHE_DIR, save_on_disk=save_on_disk,
#                                num_workers=num_workers, transform=transform,
#                                preprocessing_transform=preprocessing_transform,
#                                preprocess_at_runtime=preprocess_at_runtime)
#        if preprocess_at_runtime:
#            for _ in ds:
#                pass
#        
#        expected_files = glob.glob(os.path.join(CACHE_DIR, '*', '*.pt'))
#        files_mtime = {path: os.path.getmtime(path) for path in expected_files}
#        time.sleep(0.01) # This is just in case I/O is so fast that files get greated in the same time
#
#        # An error should be raised in the size of the dataset is different than the number of cached subfolder
#        with pytest.raises(RuntimeError,
#                           match=f"{len(dummy_dataset)} files already exist on {CACHE_DIR} this dataset as "
#                                 f"{len(larger_dataset)} files so caching is too ambiguous and error-prone "
#                                 "please force rewriting by setting 'force_overwrite'"):
#            larger_ds = ProcessedDatasetV2(larger_dataset, cache_dir=CACHE_DIR,
#                                           save_on_disk=save_on_disk,
#                                           num_workers=num_workers, transform=transform,
#                                           preprocessing_transform=preprocessing_transform,
#                                           preprocess_at_runtime=preprocess_at_runtime)
#        # Nothing got overwriten before throwing the error
#        for path, mtime in files_mtime.items():
#            assert os.path.getmtime(path) == mtime
#
#        # Check that same dataset will reuse the cached files (won't overwrite)
#        files_mtime = {path: os.path.getmtime(path) for path in expected_files}
#        larger_ds = ProcessedDatasetV2(larger_dataset, cache_dir=CACHE_DIR,
#                                       save_on_disk=save_on_disk,
#                                       num_workers=num_workers, transform=transform,
#                                       preprocessing_transform=preprocessing_transform,
#                                       force_overwrite=True,
#                                       preprocess_at_runtime=preprocess_at_runtime)
#        if preprocess_at_runtime:
#            assert len(glob.glob(os.path.join(CACHE_DIR, '*'))) == 0
#            for _ in larger_ds:
#                pass
#        
#        # with 'force_overwrite' the cache files should be overwritten
#        for path, mtime in files_mtime.items():
#            assert os.path.getmtime(path) != mtime
#
#    @pytest.mark.parametrize('preprocess_at_runtime', [True, False])
#    @pytest.mark.parametrize('use_preprocessing', [True, False])
#    @pytest.mark.parametrize('use_transform', [True, False])
#    @pytest.mark.parametrize('num_workers', [0, 4])
#    @pytest.mark.parametrize('save_on_disk', [True, ['b'], ['a', 'c']])
#    def test_fail_diff_dataset(self, dummy_dataset, diff_dataset, num_workers, save_on_disk,
#                               use_preprocessing, use_transform, preprocess_at_runtime):
#        if not use_preprocessing and isinstance(save_on_disk, list) and 'c' in save_on_disk:
#            pytest.skip("skipping because field 'c' doesn't exist without preprocessing")
#        assert not os.path.exists(CACHE_DIR)
#
#        transform = DummyTransform(10., False) if use_transform else None
#        preprocessing_transform = DummyTransform(100., True) if use_preprocessing else None
#        ds = ProcessedDatasetV2(dummy_dataset, cache_dir=CACHE_DIR, save_on_disk=save_on_disk,
#                                num_workers=num_workers, transform=transform,
#                                preprocessing_transform=preprocessing_transform,
#                                preprocess_at_runtime=preprocess_at_runtime)
#
#        if preprocess_at_runtime:
#            for _ in ds:
#                pass
#        expected_files = glob.glob(os.path.join(CACHE_DIR, '*', '*.pt'))
#        files_mtime = {path: os.path.getmtime(path) for path in expected_files}
#        time.sleep(0.01) # This is just in case I/O is so fast that files get greated in the same time
#        # An error should be raised in the size of the dataset is different than the number of cached subfolder
#        with pytest.raises(AssertionError,
#                           match=f"file '{os.path.join(CACHE_DIR, '0')}" + r"/.\.pt' is different than "
#                                 "its matching field from the input dataset, set 'force_overwriting' "
#                                 "to True to overwrite the files at preprocessing."):
#            diff_ds = ProcessedDatasetV2(diff_dataset, cache_dir=CACHE_DIR,
#                                         save_on_disk=save_on_disk,
#                                         num_workers=num_workers, transform=transform,
#                                         preprocessing_transform=preprocessing_transform,
#                                         preprocess_at_runtime=preprocess_at_runtime)
#        # Nothing got overwriten before throwing the error
#        for path, mtime in files_mtime.items():
#            assert os.path.getmtime(path) == mtime
#
#        # Check that same dataset will reuse the cached files (won't overwrite)
#        files_mtime = {path: os.path.getmtime(path) for path in expected_files}
#        diff_ds = ProcessedDatasetV2(diff_dataset, cache_dir=CACHE_DIR,
#                                     save_on_disk=save_on_disk,
#                                     num_workers=num_workers, transform=transform,
#                                     preprocessing_transform=preprocessing_transform,
#                                     force_overwrite=True,
#                                     preprocess_at_runtime=preprocess_at_runtime)
#
#        if preprocess_at_runtime:
#            for path, mtime in files_mtime.items():
#                assert os.path.getmtime(path) == mtime
#            for _ in diff_ds:
#                pass
# 
#        # with 'force_overwrite' the cache files should be overwritten
#        for path, mtime in files_mtime.items():
#            assert os.path.getmtime(path) != mtime
# 
#    @pytest.mark.parametrize('preprocess_at_runtime', [True, False])
#    @pytest.mark.parametrize('force_overwrite', [True, False])
#    @pytest.mark.parametrize('use_preprocessing', [True, False])
#    @pytest.mark.parametrize('use_transform', [True, False])
#    @pytest.mark.parametrize('num_workers', [0, 4])
#    @pytest.mark.parametrize('save_on_disk', [['d'], ['b', 'd']])
#    def test_fail_wrong_field(self, dummy_dataset, num_workers,
#                              use_preprocessing, use_transform,
#                              force_overwrite, save_on_disk, preprocess_at_runtime):
#        assert not os.path.exists(CACHE_DIR)
#
#        transform = DummyTransform(10., False) if use_transform else None
#        preprocessing_transform = DummyTransform(100., True) if use_preprocessing else None
#        with pytest.raises(ValueError,
#                           match="the dataset doesn't provide an output field 'd'"):
#            ds = ProcessedDatasetV2(dummy_dataset, cache_dir=CACHE_DIR, save_on_disk=save_on_disk,
#                                    num_workers=num_workers, transform=transform,
#                                    preprocessing_transform=preprocessing_transform,
#                                    force_overwrite=force_overwrite,
#                                    preprocess_at_runtime=preprocess_at_runtime)
#
#
#    @pytest.mark.parametrize('preprocess_at_runtime', [True, False])
#    @pytest.mark.parametrize('use_transform', [True, False])
#    @pytest.mark.parametrize('num_workers', [0, 4])
#    def test_more_fields(self, dummy_dataset, num_workers, use_transform,
#                         preprocess_at_runtime):
#        assert not os.path.exists(CACHE_DIR)
#
#        transform = DummyTransform(10., False) if use_transform else None
#        ds = ProcessedDatasetV2(dummy_dataset, cache_dir=CACHE_DIR, save_on_disk=True,
#                                num_workers=num_workers, transform=transform,
#                                preprocessing_transform=DummyTransform(100., False),
#                                preprocess_at_runtime=preprocess_at_runtime)
#        if preprocess_at_runtime:
#            for _ in ds:
#                pass
#        expected_files = glob.glob(os.path.join(CACHE_DIR, '*', '*.pt'))
#        files_mtime = {path: os.path.getmtime(path) for path in expected_files}
#
#        ds2 = ProcessedDatasetV2(dummy_dataset, cache_dir=CACHE_DIR, save_on_disk=True,
#                                 num_workers=num_workers, transform=transform,
#                                 preprocessing_transform=DummyTransform(100., True),
#                                 preprocess_at_runtime=preprocess_at_runtime)
#        if preprocess_at_runtime:
#            for _ in ds:
#                pass
#
#        for path, mtime in files_mtime.items():
#            assert os.path.getmtime(path) == mtime
#
#    @pytest.mark.parametrize('force_overwrite', [True, False])
#    @pytest.mark.parametrize('preprocess_at_runtime', [True, False])
#    @pytest.mark.parametrize('use_transform', [True, False])
#    @pytest.mark.parametrize('num_workers', [0, 4])
#    def test_less_fields(self, dummy_dataset, num_workers, use_transform,
#                         force_overwrite, preprocess_at_runtime):
#        """If we remove a field to be saved, we should reuse the cache"""
#        assert not os.path.exists(CACHE_DIR)
#
#        transform = DummyTransform(10., False) if use_transform else None
#        ds = ProcessedDatasetV2(dummy_dataset, cache_dir=CACHE_DIR, save_on_disk=True,
#                                num_workers=num_workers, transform=transform,
#                                preprocessing_transform=DummyTransform(100., True))
#
#        expected_files = glob.glob(os.path.join(CACHE_DIR, '*', '*.pt'))
#        files_mtime = {path: os.path.getmtime(path) for path in expected_files}
#
#        ds2 = ProcessedDatasetV2(dummy_dataset, cache_dir=CACHE_DIR, save_on_disk=True,
#                                 num_workers=num_workers, transform=transform,
#                                 preprocessing_transform=DummyTransform(100., False),
#                                 preprocess_at_runtime=preprocess_at_runtime,
#                                 force_overwrite=force_overwrite)
#        if preprocess_at_runtime:
#            # with preproces_at_runtime files are created after using __getitem__
#            for _ in ds2:
#                pass
#        time.sleep(0.01) # This is just in case I/O is so fast that files get greated in the same time
#        if force_overwrite:
#            # All the files but the fields that have been remove should be replaced
#            kept_files = glob.glob(os.path.join(CACHE_DIR, '*', 'c.pt'))
#            for path, mtime in files_mtime.items():
#                if path in kept_files:
#                    assert os.path.getmtime(path) == mtime
#                else:
#                    assert os.path.getmtime(path) != mtime
#        else:
#            for path, mtime in files_mtime.items():
#                assert os.path.getmtime(path) == mtime
#
#    @pytest.mark.parametrize('preprocess_at_runtime', [True, False])
#    @pytest.mark.parametrize('use_preprocessing', [True, False])
#    @pytest.mark.parametrize('use_transform', [True, False])
#    @pytest.mark.parametrize('num_workers', [0, 4])
#    @pytest.mark.parametrize('force_overwrite', [True, False])
#    @pytest.mark.parametrize('save_on_disk', [True, ['b'], ['a', 'c']])
#    def test_cleaning_on_fail(self, dummy_dataset, num_workers, use_preprocessing,
#                              use_transform, save_on_disk, force_overwrite, preprocess_at_runtime):
#        if not use_preprocessing and isinstance(save_on_disk, list) and 'c' in save_on_disk:
#            pytest.skip("skipping because field 'c' doesn't exist without preprocessing")
#        assert not os.path.exists(CACHE_DIR)
#        transform = DummyTransform(10., False) if use_transform else None
#        preprocessing_transform = DummyTransform(100., True) if use_preprocessing else None
#
#        dummy_dataset.threshold = 5
#
#        if preprocess_at_runtime:
#            ds = ProcessedDatasetV2(dummy_dataset, cache_dir=CACHE_DIR, save_on_disk=save_on_disk,
#                                    num_workers=num_workers, transform=transform,
#                                    preprocessing_transform=preprocessing_transform,
#                                    preprocess_at_runtime=preprocess_at_runtime,
#                                    force_overwrite=force_overwrite)
#
#            for i in range(5):
#                _ = ds[i]
#                assert len(glob.glob(os.path.join(CACHE_DIR, str(i), '*.pt'))) > 0
#            with pytest.raises(RuntimeError, match='this is an error.'):
#                _ = ds[5]
#        else:
#            with pytest.raises(RuntimeError, match='this is an error.'):
#                ds = ProcessedDatasetV2(
#                    dummy_dataset, cache_dir=CACHE_DIR, save_on_disk=save_on_disk,
#                    num_workers=num_workers, transform=transform,
#                    preprocessing_transform=preprocessing_transform,
#                    preprocess_at_runtime=preprocess_at_runtime,
#                    force_overwrite=force_overwrite)
#
#        assert not os.path.exists(CACHE_DIR)

### DEPRECATED ###
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
