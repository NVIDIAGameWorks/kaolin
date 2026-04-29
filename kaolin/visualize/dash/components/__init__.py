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

# Note: we assume that dash is available, as this is checked by parent init
import os as _os
import dash as _plotly_dash

# noinspection PyUnresolvedReferences
# TODO: is noinspection needed?
from .autogen._imports_ import *
from .autogen._imports_ import __all__

__all__ = [k for k in locals().keys() if not k.startswith('_')]

_js_dist = []
_js_dist.extend(
    [
        {
            'relative_package_path': 'kaolin.js',
            'namespace': 'kaolin'
        },
        {
            'relative_package_path': 'kaolin.js.map',
            'namespace': 'kaolin',
            'dynamic': True
        },
        {
            'dev_package_path': 'proptypes.js',
            'namespace': 'kaolin',
            'dev_only': True,
        }
    ]
)

_css_dist = []

for _component in __all__:
    setattr(locals()[_component], '_js_dist', _js_dist)
    setattr(locals()[_component], '_css_dist', _css_dist)
