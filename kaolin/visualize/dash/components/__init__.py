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
