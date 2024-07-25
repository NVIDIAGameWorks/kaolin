.. _diff_render:

Differentiable Rendering
************************

.. image:: ../img/clock.gif

Differentiable rendering can be used to optimize the underlying 3D properties, like geometry and lighting, by back-propagating gradients from the loss in the image space. Kaolin Library integrates techniques from `DIB-R <https://research.nvidia.com/labs/toronto-ai/DIB-R/>`_ and `DIB-R++ <https://research.nvidia.com/labs/toronto-ai/DIBRPlus/>`_ published techniques, as well as follow up improvements. Many Kaolin utilities also integrate with an alternative `nvdiffrast <https://github.com/NVlabs/nvdiffrast>`_ differentiable rendering utility, also from NVIDIA.

We provide an end-to-end basic tutorial using the :mod:`kaolin.render.mesh` functionality for mesh optimization in `examples/tutorial/dibr_tutorial.ipynb <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/dibr_tutorial.ipynb>`_. See also :ref:`Differentiable Camera <differentiable_camera>`, :ref:`Differentiable Lighting <differentiable_lighting>`, and :ref:`Easy PBR Shader <pbr_shader>`.


References
==========

* DIB-R: `"Learning to predict 3d objects with an interpolation-based differentiable renderer." <https://research.nvidia.com/labs/toronto-ai/DIB-R/>`_ Chen, Wenzheng, Huan Ling, Jun Gao, Edward Smith, Jaakko Lehtinen, Alec Jacobson, and Sanja Fidler. NeurIPS 2019.
* `"DIB-R++: learning to predict lighting and material with a hybrid differentiable renderer." <https://research.nvidia.com/labs/toronto-ai/DIBRPlus/>`_ Chen, Wenzheng, Joey Litalien, Jun Gao, Zian Wang, Clement Fuji Tsang, Sameh Khamis, Or Litany, and Sanja Fidler. NeurIPS 2021.
* Nvdiffrast: `"Modular primitives for high-performance differentiable rendering." <https://github.com/NVlabs/nvdiffrast>`_ Laine, Samuli, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, and Timo Aila.  SIGGRAPH (TOG) 2020.
