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

# TODO: remove if not needed
#_current_path = os.path.dirname(os.path.abspath(__file__))
#_this_module = sys.modules[__name__]

from .assets_helper import *
from .builder import *
from .builtins import *
from .layout import *
from . import option
from . import user_behavior_scan
from .viewer import *


def _print_warn():
    print('Plotly Dash not available. Kaolin Dash functionality will not be available.')


_has_plotly_dash = False
try:
    import dash as _plotly_dash
    _has_plotly_dash = True
    # TODO: maybe there should also be another condition here? We don't need to import all.
except ImportError:
    _print_warn()


def plotly_dash_is_available():
    """Returns True if plotly dash is available, False otherwise."""
    if not _has_plotly_dash:
        _print_warn()
    return _has_plotly_dash


# Only import Dash components if available
if _has_plotly_dash:
    # noinspection PyUnresolvedReferences
    # TODO: is noinspection needed?
    from . import components

__all__ = [k for k in locals().keys() if not k.startswith('_')]
