from .shapenet import *
from .modelnet import *
from .shrec import *
from .scannet import *

# nuscenes-devkit will import matplotlib trying for an x11 backend, workaround here
import matplotlib
matplotlib.use('Agg')

try:
    from .nusc import NuscDetection
except ImportError as err:
    import_err = err
    import traceback
    print("Warning: unable to import datasets/nusc:\n   %s" % import_err)
    print("Warning: unable to import datasets/nusc:\n   %s" % traceback.print_exc())
