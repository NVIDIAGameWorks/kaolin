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
import sys
import pytest
import torch
from kaolin.utils.log import default_log_setup

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
