from . import io
from . import math
from . import metrics
from . import ops
from . import render
from . import rep
from . import utils
from . import visualize
from . import physics
from . import non_commercial

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass


# HACK (maybe? allows dash to actually serve kaolin javascript code)
import os as _os
__current_path = _os.path.dirname(_os.path.abspath(__file__))
#__dash_js_dir = _os.path.join(__current_path, 'visualize', 'dash', 'components', 'autogen')
__dash_js_dir = _os.path.join('visualize', 'dash', 'components', 'autogen')
if _os.path.exists(_os.path.join(__current_path, __dash_js_dir, 'kaolin.js')):
    # Note: _js_dist is a special module-level variable that Plotly Dash looks for to find
    # javascript distributions for custom *python* Dash components. This makes sense, because there
    # is typically one javascript file for all the components within a single module, as is the
    # case for Kaolin as well. 
    # Debug notes:
    # If running into issues, step through the code in the PlotlyDash ComponentRegistry.
    _js_dist = []
    _js_dist.extend(
        [
            {
                'relative_package_path': _os.path.join(__dash_js_dir, 'kaolin.js'), 
                'namespace': 'kaolin'
            },
            {
                'relative_package_path': _os.path.join(__dash_js_dir, 'kaolin.js.map'),
                'namespace': 'kaolin',
                'dynamic': True
            },
            {
                'dev_package_path': _os.path.join(__dash_js_dir, 'prototypes.js'),
                'namespace': 'kaolin',
                'dev_only': True,
            }
        ]
    )

    _css_dist = []