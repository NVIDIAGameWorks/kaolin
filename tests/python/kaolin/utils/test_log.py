# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import re
import sys
import pytest
import torch
from kaolin.utils.log import default_log_setup, setup_log_file

from kaolin.utils import log

logger = logging.getLogger(__name__)


def remove_log_setup():
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    old_level = root.level

    for h in root.handlers[:]:
        root.removeHandler(h)

    return {'handlers': old_handlers, 'level': old_level}


def restore_log_setup(handlers, level):
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)
    root.setLevel(level)


class TestDefaultLogSetup:

    # Note: using lines from a poem by E.E. Cummings
    def test_no_log_without_setup(self, caplog):
        prev_setup = remove_log_setup()

        # Sanity test that the default condition does not produce log
        with caplog.at_level(logging.DEBUG):
            logging.log(logging.INFO, 'anyone lived in a pretty how town')
            assert len(caplog.records) == 0
        with caplog.at_level(logging.DEBUG):
            logger.log(logging.INFO, '(with up so floating many bells down)')
            assert len(caplog.records) == 0

        restore_log_setup(**prev_setup)

    @pytest.mark.parametrize('use_root', [True, False])
    @pytest.mark.parametrize('level', [logging.DEBUG, logging.INFO])
    def test_has_log_with_setup(self, level, use_root, capsys):
        prev_setup = remove_log_setup()

        default_log_setup(level)

        line1 = 'spring summer autumn winter'
        line2 = 'he sang his didn’t he danced his did.'
        use_logger = logging if use_root else logger  # Test using root logger or module logger


        use_logger.log(logging.INFO, line1)
        use_logger.log(logging.DEBUG, line2)
        captured = capsys.readouterr()
        assert line1 in captured.out
        if level == logging.DEBUG:
            assert line2 in captured.out

        restore_log_setup(**prev_setup)


class TestAddLogLevelFlag:
    def test_adds_flag_with_default_info(self):
        parser = argparse.ArgumentParser()
        log.add_log_level_flag(parser)
        args = parser.parse_args([])
        assert args.log_level == logging.INFO

    def test_parses_log_level_arg(self):
        parser = argparse.ArgumentParser()
        log.add_log_level_flag(parser)
        args = parser.parse_args(['--log_level', '10'])
        assert args.log_level == 10


class TestLogTensor:
    def test_log_tensor_emits_message(self, caplog):
        t = torch.randn(2, 3)
        logger = logging.getLogger('test_log_tensor')
        with caplog.at_level(logging.DEBUG, logger='test_log_tensor'):
            log.log_tensor(t, 'my_tensor', use_logger=logger, level=logging.DEBUG)
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.DEBUG
        assert 'my_tensor' in caplog.records[0].message
        assert 'torch.Size' in caplog.records[0].message or '2' in caplog.records[0].message


class TestSetupLogFile:
    """Tests for :func:`kaolin.utils.log.setup_log_file`."""

    @pytest.fixture(autouse=True)
    def _isolate_root_handlers(self):
        """Close and remove any FileHandlers added during the test."""
        root = logging.getLogger()
        handlers_before = set(root.handlers)
        level_before = root.level
        yield
        for h in root.handlers[:]:
            if h not in handlers_before:
                h.close()
                root.removeHandler(h)
        root.setLevel(level_before)

    def test_returns_absolute_path_inside_log_dir(self, tmp_path):
        path = setup_log_file(str(tmp_path), prefix='test')
        assert os.path.isabs(path)
        assert os.path.dirname(path) == str(tmp_path)

    def test_filename_matches_prefix_and_timestamp_pattern(self, tmp_path):
        path = setup_log_file(str(tmp_path), prefix='myapp')
        filename = os.path.basename(path)
        assert re.match(r'^myapp_\d{4}-\d{2}-\d{2}_\d{6}\.log$', filename), \
            f'Unexpected filename: {filename}'

    def test_default_prefix_is_kaolin(self, tmp_path):
        path = setup_log_file(str(tmp_path))
        assert os.path.basename(path).startswith('kaolin_')

    def test_log_file_is_created(self, tmp_path):
        path = setup_log_file(str(tmp_path), prefix='test')
        assert os.path.isfile(path)

    def test_creates_log_dir_if_absent(self, tmp_path):
        new_dir = tmp_path / 'nested' / 'logs'
        assert not new_dir.exists()
        setup_log_file(str(new_dir), prefix='test')
        assert new_dir.is_dir()

    def test_attaches_file_handler_to_root_logger(self, tmp_path):
        root = logging.getLogger()
        file_handlers_before = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        setup_log_file(str(tmp_path), prefix='test')
        file_handlers_after = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers_after) == len(file_handlers_before) + 1

    def test_messages_written_to_file(self, tmp_path):
        path = setup_log_file(str(tmp_path), prefix='test')
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        logging.getLogger('test_messages').info('hello from setup_log_file test')
        for h in root.handlers:
            h.flush()
        assert 'hello from setup_log_file test' in open(path).read()

    def test_file_format_matches_standard_kaolin_format(self, tmp_path):
        path = setup_log_file(str(tmp_path), prefix='test')
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        logging.getLogger('test_fmt').info('format check')
        for h in root.handlers:
            h.flush()
        content = open(path).read()
        # Expected: 2026-08-13 12:00:00,000|    INFO|       test_fmt| format check
        assert re.search(r'\d{4}-\d{2}-\d{2}.*\|\s*INFO\|.*\| format check', content)

    def test_no_duplicate_handler_for_same_path(self, tmp_path):
        """Calling setup_log_file twice with a path that is already registered
        as a FileHandler on the root logger must not add a second handler."""
        import unittest.mock as mock

        known_path = str(tmp_path / 'fixed.log')
        # Pre-register a FileHandler for that exact path.
        existing = logging.FileHandler(known_path)
        logging.getLogger().addHandler(existing)
        try:
            # Patch the internals so setup_log_file resolves to the same known_path.
            with mock.patch('datetime.datetime') as mock_dt, \
                 mock.patch('os.makedirs'), \
                 mock.patch('os.path.abspath', return_value=known_path):
                mock_dt.now.return_value.strftime.return_value = 'fixed'
                result = setup_log_file(str(tmp_path), prefix='fixed')

            assert result == known_path
            count = sum(
                1 for h in logging.getLogger().handlers
                if isinstance(h, logging.FileHandler)
                and os.path.abspath(h.baseFilename) == known_path
            )
            assert count == 1, f'Expected 1 FileHandler for {known_path}, got {count}'
        finally:
            existing.close()
            logging.getLogger().removeHandler(existing)


class TestPrintTensor:
    def test_print_tensor_stdout(self, capsys):
        t = torch.tensor([1.0, 2.0, 3.0])
        log.print_tensor(t, 'vec')
        out, _ = capsys.readouterr()
        assert 'vec' in out
        assert 'torch.Size' in out or '3' in out

    def test_print_tensor_with_stats(self, capsys):
        t = torch.tensor([1.0, 2.0, 3.0])
        log.print_tensor(t, 'x', print_stats=True)
        out, _ = capsys.readouterr()
        assert 'x' in out
        assert 'min' in out or 'max' in out or 'mean' in out
