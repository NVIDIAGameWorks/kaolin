from .shapenet import *
from .modelnet import *
from .shrec import *
from .scannet import *
try:
    from .nusc import NuscDetection
except ImportError as err:
    import_err = err
    print("Warning: unable to import datasets/nusc:\n   %s" % import_err)
