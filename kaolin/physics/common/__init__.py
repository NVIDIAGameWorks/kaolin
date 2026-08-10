r"""Building blocks shared by the physics simulators: forces, contact, and solvers.

Nothing here depends on Simplicits. The forces act on plain ``wp.array`` positions, and
:func:`newtons_method_capturable` takes callbacks, so both can be driven directly with no
scene object.

.. _scene_forces_capture:

Using the forces inside a CUDA graph
------------------------------------

A recorded graph stores raw device addresses and the Python values that were in scope
when it was recorded. Replaying it re-runs those launches; it does not re-run any Python.
Three consequences, none of which raises if you get it wrong:

* **Settings are frozen at record time.** :attr:`Gravity.g`, ``Floor.floor_height``,
  ``Floor.floor_axis``, ``Floor.flip_floor`` and the ``coeff`` argument are read on the
  host and compiled in. Changing one after recording has no effect on the replay --
  verified by moving a floor from 0.5 to -100.0 and watching the replayed energy not
  budge. To change any of them, record the graph again.
* **Pass output buffers in.** ``energy(...)`` and ``gradient(...)`` allocate when their
  output argument is left as ``None``, which a capture forbids; the ``energy`` fallback
  also lands on Warp's *default* device rather than the one the inputs live on. Inside a
  recording, always pass a preallocated buffer.
* **Any array handed in from outside must keep its address.** The graph refers to the
  buffer that was there when it was recorded, not to whatever a variable points at now.
  Where an array comes from a library that can either allocate a fresh one or fill one in
  place, use the in-place form -- for example allocate a contacts buffer once and pass it
  to every collision query, rather than letting each query hand back a new one.

Two behaviours that surprise people whether or not a graph is involved:

* ``energy`` and ``gradient`` **accumulate** into their output and never zero it, so
  several forces can sum into one buffer. Zeroing is the caller's job; calling twice
  without zeroing doubles the result.
* ``hessian`` returns a buffer **owned by the force**, and the next call to that force
  overwrites it. Consume the result before calling again, or take a copy.

The equivalent rules for the solver's callbacks are in
:func:`newtons_method_capturable`.
"""

from .collisions import *
from .optimization import *
from .optimization_capturable import *
from .scene_forces import *

__all__ = [k for k in locals().keys() if not k.startswith('__')]
