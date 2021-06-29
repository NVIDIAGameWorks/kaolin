.. _kaolin.ops.mesh:

kaolin.ops.mesh
---------------

A mesh is a 3D object representation consisting of a collection of vertices and polygons. In Kaolin, vertices are of
shape :math:`(\text{batch_size}, \text{num_vertices}, 3)` and faces are of shape :math:`(\text{num_faces}, 3)`. 

Triangular meshes
==================
Triangular meshes comprise of a set of triangles that are connected by their common edges or corners. 

Tetrahedral meshes
==================

A tetrahedron or triangular pyramid is a polyhedron composed of four triangular faces, six straight edges, and four
vertex corners. Tetrahedral meshes inside Kaolin are composed of tetrahedrons which have 4 vertices and each vertex has
3 dimensions. Hence, their shape is :math:`(\text{batch_size}, \text{num_tetrahedrons}, 4, 3)`.

API
---

.. automodule:: kaolin.ops.mesh
   :members:
   :undoc-members:
   :show-inheritance:
