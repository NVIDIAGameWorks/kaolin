.. _kaolin.ops.mesh:

kaolin.ops.mesh
***********************

A mesh is a 3D object representation consisting of a collection of vertices and polygons.

Triangular meshes
==================

Triangular meshes comprise of a set of triangles that are connected by their common edges or corners. In Kaolin, they are usually represented as a set of two tensors:

* ``vertices``: A :class:`torch.Tensor`, of shape :math:`(\text{batch_size}, \text{num_vertices}, 3)`, contains the vertices coordinates.

* ``faces``: A :class:`torch.LongTensor`, of shape :math:`(\text{batch_size}, \text{num_faces}, 3)`, contains the mesh topology, by listing the vertices index for each face.

Both tensors can be combined using :func:`kaolin.ops.mesh.index_vertices_by_faces`, to form ``face_vertices``, of shape :math:`(\text{batch_size}, \text{num_faces}, 3, 3)`, listing the vertices coordinate for each face.


Tetrahedral meshes
==================

A tetrahedron or triangular pyramid is a polyhedron composed of four triangular faces, six straight edges, and four vertex corners. Tetrahedral meshes inside Kaolin are composed of two tensors:

* ``vertices``: A :class:`torch.Tensor`, of shape :math:`(\text{batch_size}, \text{num_vertices}, 3)`, contains the vertices coordinates.

* ``tet``: A :class:`torch.LongTensor`, of shape :math:`(\text{batch_size}, \text{num_tet}, 4)`, contains the tetrahedral mesh topology, by listing the vertices index for each tetrahedron.

Both tensors can be combined, to form ``tet_vertices``, of shape :math:`(\text{batch_size}, \text{num_tet}, 4, 3)`, listing the tetrahedrons vertices coordinates for each face.


API
---

.. automodule:: kaolin.ops.mesh
   :members:
   :undoc-members:
   :show-inheritance:
