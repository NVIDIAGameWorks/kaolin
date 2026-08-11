r"""Physics helpers for forces, contact, and solving.

The forces work on plain ``wp.array`` positions. The captured Newton solver takes
callbacks, so neither requires a Simplicits scene.

.. _scene_forces_capture:

Using the forces inside a CUDA graph
------------------------------------

Before recording a graph, set force options and allocate output buffers. To change an
option or replace an array later, record a new graph.

``energy`` and ``gradient`` add to their output; clear that output before reuse.
``hessian`` returns a buffer owned by the force and overwrites it on the next call.
See :func:`newtons_method_capturable` for the solver rules.
"""

from .collisions import *
from .optimization import *
from .optimization_capturable import *
from .scene_forces import *

__all__ = [k for k in locals().keys() if not k.startswith('__')]
