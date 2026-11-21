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
import logging as _logging
import sys as _sys
import torch as _torch

from .testing import tensor_info


_logger = _logging.getLogger(__name__)

__all__ = ['default_log_setup', 'add_log_level_flag', 'log_tensor', 'print_tensor', 'setup_log_file']


def default_log_setup(level=_logging.INFO, force=True):
    """Set up default logging to stdout and quiet optional dependencies.

    Args:
        level (int): Logging level for the root logger (default: ``logging.INFO``, i.e. 20).
        force (bool): if True (default), will replace any existing loggers with simple stdout.
    """
    root = _logging.getLogger()
    handlers = [_logging.StreamHandler(_sys.stdout)]
    _logging.basicConfig(
        level=level,
        format="%(asctime)s|%(levelname)8s|%(name)15s| %(message)s",
        handlers=handlers,
        force=force) # force even if root has loggers already
    root.setLevel(level)
    _logger.info("Logging to stdout")
    _torch.set_printoptions(linewidth=120)

    # Reduce noise from optional dependencies in DEBUG mode; ignore if not installed.
    dependency_default_level = _logging.INFO
    if level < dependency_default_level:  # debug mode
        try:
            _logging.getLogger("PIL.PngImagePlugin").setLevel(dependency_default_level)
            _logging.getLogger("PIL.Image").setLevel(dependency_default_level)
            _logging.getLogger("PIL").setLevel(dependency_default_level)
        except Exception:
            pass
        try:
            _logging.getLogger("matplotlib.font_manager").setLevel(dependency_default_level)
            _logging.getLogger("matplotlib.axes._base").setLevel(dependency_default_level)
            _logging.getLogger("matplotlib.pyplot").setLevel(dependency_default_level)
        except Exception:
            pass


def add_log_level_flag(parser):
    """
    Add a log_level flag to an argparser.

    Args:
        parser (ArgumentParser): The argparser to add the flag to.
    """

    parser.add_argument(
        "--log_level",
        action="store",
        type=int,
        default=_logging.INFO,
        help="Logging level to use globally, DEBUG: 10, INFO: 20, WARN: 30, ERROR: 40.",
    )


def log_tensor(t, name, use_logger=None, level=_logging.DEBUG, print_stats=False, detailed=False):
    """Log diagnostic tensor information (shape, dtype, optional stats) via a logger.

    Uses :func:`~kaolin.utils.testing.tensor_info` to format the message.

    Args:
        t (torch.Tensor or numpy.ndarray or None): The tensor to describe.
        name (str): Human-readable name for the tensor in the log message.
        use_logger (logging.Logger, optional): Logger to use. Default: the module logger.
        level (int): Logging level (default: ``logging.DEBUG``, i.e. 10).
        print_stats (bool): If True, include min/max/mean in the message (default: False).
        detailed (bool): If True, include extra tensor properties (default: False).

    Examples:
        >>> t = torch.tensor([1., 2., 3.])
        >>> log_tensor(t, 'my_tensor', level=logging.INFO)
    """
    if use_logger is None:
        use_logger = _logger

    use_logger.log(level, tensor_info(t, name, print_stats=print_stats, detailed=detailed))


def setup_log_file(log_dir: str, prefix: str = 'kaolin') -> str:
    """Attach a :class:`logging.FileHandler` to the root logger and return the log file path.

    The file is created immediately inside *log_dir* (which is made if it does not
    already exist) and shares the same format as :func:`default_log_setup`::

        %(asctime)s|%(levelname)8s|%(name)15s| %(message)s

    This is a complement to :func:`default_log_setup`, not a replacement — call
    both when you want logs on both stdout and in a file.

    Args:
        log_dir (str): Directory in which the log file will be created.
        prefix (str): Filename prefix (default: ``'kaolin'``).  The final name
            is ``<prefix>_<YYYY-MM-DD_HHMMSS>.log``.

    Returns:
        str: Absolute path of the created log file.

    Examples:
        >>> path = setup_log_file('/tmp/kaolin_logs', prefix='segment')
        >>> # '/tmp/kaolin_logs/segment_2026-08-04_153000.log'
    """
    import os as _os
    from datetime import datetime as _datetime

    _os.makedirs(log_dir, exist_ok=True)
    timestamp = _datetime.now().strftime('%Y-%m-%d_%H%M%S')
    log_path = _os.path.abspath(_os.path.join(log_dir, f'{prefix}_{timestamp}.log'))

    root = _logging.getLogger()
    if any(isinstance(h, _logging.FileHandler) and
           _os.path.abspath(h.baseFilename) == log_path
           for h in root.handlers):
        _logger.debug(f'FileHandler for {log_path} already attached; skipping duplicate.')
        return log_path

    file_handler = _logging.FileHandler(log_path)
    file_handler.setFormatter(_logging.Formatter(
        '%(asctime)s|%(levelname)8s|%(name)15s| %(message)s'))
    root.addHandler(file_handler)

    _logger.info(f'Logging to file: {log_path}')
    return log_path


def print_tensor(t, name, print_stats=False, detailed=False):
    """Print diagnostic tensor information (shape, dtype, optional stats) to stdout.

    Uses :func:`~kaolin.utils.testing.tensor_info` to format the message.

    Args:
        t (torch.Tensor or numpy.ndarray or None): The tensor to describe.
        name (str): Human-readable name for the tensor in the output.
        print_stats (bool): If True, include min/max/mean (default: False).
        detailed (bool): If True, include extra tensor properties (default: False).

    Examples:
        >>> t = torch.tensor([1., 2., 3.])
        >>> print_tensor(t, 'my_tensor')
        my_tensor: torch.Size([3]) (torch.float32)
    """
    print(tensor_info(t, name, print_stats=print_stats, detailed=detailed))
