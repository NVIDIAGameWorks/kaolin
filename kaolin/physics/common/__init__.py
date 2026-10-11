r"""Physics helpers for forces, contact, and solving.

The forces work on plain ``wp.array`` positions. The captured Newton solver takes
callbacks, so neither requires a Simplicits scene.

.. _scene_forces_capture:

Using the forces inside a CUDA graph
------------------------------------

Before recording a graph, set force options and allocate output buffers. To change an
option or replace an array later, record a new graph.

For :class:`Gravity`, :class:`Floor` and :class:`Boundary`: ``energy`` and ``gradient``
add to their output, so several forces can sum into one buffer and clearing it is the
caller's job; ``hessian`` returns a buffer owned by the force and overwrites it on the
next call, so consume the result before calling again.

:class:`Collision` does **not** follow that rule and cannot be summed into a shared
buffer alongside the others: ``gradient`` clears the output it is given and its kernel
assigns rather than accumulates, so whatever the other forces contributed is lost, and
``hessian`` hands back the buffer it was passed rather than one of its own. Give it its
own output and add that in yourself. Its ``energy`` does accumulate, so the class is not
internally consistent either.

See :func:`newtons_method_capturable` for the solver rules.
"""

from .collisions import *
from .optimization import *
from .optimization_capturable import *
from .scene_forces import *

__all__ = [k for k in locals().keys() if not k.startswith('__')]
