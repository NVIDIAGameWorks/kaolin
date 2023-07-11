:orphan:

.. _kaolin.rep.SurfaceMesh:

SurfaceMesh
===========================

Tutorial
--------

For a walk-through of :class:`kaolin.rep.SurfaceMesh` features,
see `working_with_meshes.ipynb <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/working_with_meshes.ipynb>`_.

API
---

* :ref:`Overview <rubric mesh overview>`
* :ref:`Supported Attributes <rubric mesh attributes>`
* :ref:`Batching <rubric mesh batching>`
* :ref:`Attribute Access and Auto-Computability <rubric mesh attribute access>`
* :ref:`Inspecting and Copying <rubric mesh inspecting>`
* :ref:`Tensor Operations <rubric mesh tensor ops>`

.. autoclass:: kaolin.rep.SurfaceMesh
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: Batching, attribute_info_string, set_batching, to_batched, getattr_batched, cat,
      vertices, face_vertices, normals, face_normals, vertex_normals, uvs, face_uvs, faces, face_normals_idx, face_uvs_idx,
      material_assignments, materials, cuda, cpu, to, float_tensors_to, detach, get_attributes, has_attribute, has_or_can_compute_attribute,
      probably_can_compute_attribute, get_attribute, get_or_compute_attribute, check_sanity, to_string, as_dict, describe_attribute,
      unset_attributes_return_none, allow_auto_compute, batching, convert_attribute_batching


   .. _rubric mesh batching:

   .. rubric:: Supported Batching Strategies

   ``SurfaceMesh`` can be instantiated with any of the following batching
   strategies, and supports conversions between batching strategies. Current
   batching strategy of a ``mesh`` object can be read from ``mesh.batching`` or
   by running ``print(mesh)``.

   For example::

       mesh = kaolin.io.obj.load_mesh(path)
       print(mesh)
       mesh.to_batched()
       print(mesh)

   .. autoclass:: kaolin.rep.SurfaceMesh.Batching
      :members:

   .. automethod:: attribute_info_string
   .. automethod:: check_sanity
   .. automethod:: set_batching
   .. automethod:: to_batched
   .. automethod:: getattr_batched
   .. automethod:: cat
   .. automethod:: convert_attribute_batching

   .. _rubric mesh attribute access:

   .. rubric:: Attribute Access

   By default, ``SurfaceMesh`` will attempt to auto-compute missing attributes
   on access. These attributes will be cached, unless their ancestors have
   ``requires_grad == True``. This behavior of the ``mesh`` object can be changed
   at construction time (``allow_auto_compute=False``) or by setting
   ``mesh.allow_auto_compute`` later. In addition to this convenience API,
   explicit methods for attribute access are also supported.

   For example, using **convenience API**::

       # Caching is enabled by default
       mesh = kaolin.io.obj.load_mesh(path, with_normals=False)
       print(mesh)
       print(mesh.has_attribute('face_normals'))  # False
       fnorm = mesh.face_normals  # Auto-computed
       print(mesh.has_attribute('face_normals'))  # True (cached)

       # Caching is disabled when gradients need to flow
       mesh = kaolin.io.obj.load_mesh(path, with_normals=False)
       mesh.vertices.requires_grad = True   # causes caching to be off
       print(mesh.has_attribute('face_normals'))  # False
       fnorm = mesh.face_normals  # Auto-computed
       print(mesh.has_attribute('face_normals'))  # False (caching disabled)


   For example, using **explicit API**::

       mesh = kaolin.io.obj.load_mesh(path, with_normals=False)
       print(mesh.has_attribute('face_normals'))  # False
       fnorm = mesh.get_or_compute_attribute('face_normals', should_cache=False)
       print(mesh.has_attribute('face_normals'))  # False


   .. automethod:: get_attributes
   .. automethod:: has_attribute
   .. automethod:: has_or_can_compute_attribute
   .. automethod:: probably_can_compute_attribute
   .. automethod:: get_attribute
   .. automethod:: get_or_compute_attribute

   .. _rubric mesh inspecting:

   .. rubric:: Inspecting and Copying Meshes

   To make it easier to work with, ``SurfaceMesh`` supports detailed print
   statements, as well as ``len()``, ``copy()``, ``deepcopy()`` and can be converted
   to a dictionary.

   Supported operations::

       import copy
       mesh_copy = copy.copy(mesh)
       mesh_copy = copy.deepcopy(mesh)
       batch_size = len(mesh)

       # Print default attributes
       print(mesh)

       # Print more detailed attributes
       print(mesh.to_string(detailed=True, print_stats=True))

       # Print specific attribute
       print(mesh.describe_attribute('vertices'))

   .. automethod:: to_string
   .. automethod:: describe_attribute
   .. automethod:: as_dict

   .. _rubric mesh tensor ops:

   .. rubric:: Tensor Operations

   Convenience operations for device and type conversions of some or all member
   tensors.

   .. automethod:: cuda
   .. automethod:: cpu
   .. automethod:: to
   .. automethod:: float_tensors_to
   .. automethod:: detach

   .. rubric:: Other
