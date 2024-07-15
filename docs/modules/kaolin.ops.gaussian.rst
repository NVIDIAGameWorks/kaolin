.. _kaolin.ops.gaussian:

kaolin.ops.gaussian
***********************

`Gaussian Splats <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>`_ is a novel 3D representation consisting
of a collection of optimizable 3D Gaussian particles carrying values like alpha and radiance.

kaolin does not address specifics regarding the optimization and rendering of this representation,
which are already addressed by other frameworks.
Rather, it supports additional novel operations which are not handled by other common packages.

To maintain compatibility with other frameworks in the broader sense, kaolin makes minimal assumptions
about the exact fields tracked by the 3D Gaussians. At the bare minimum, gaussians are expected
to keep track of their mean, covariance rotation & scale components, and opacity.


Densification
==================

The marriage of high quality reconstructions available with
`Gaussian Splats <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>`_
and physics simulations now supported by kaolin paves the way to new and exciting interactive opportunities.

To improve the accuracy of such simulations, kaolin includes a CUDA based densification module which
attempts to sample additional points within the volume of shapes represented with 3D Gaussians.

API
---

.. automodule:: kaolin.ops.gaussian
   :members:
   :undoc-members:
   :show-inheritance:
