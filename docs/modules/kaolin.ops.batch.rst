.. _kaolin.ops.batch:

kaolin.ops.batch
================

.. _batching:

Batching
--------

Batching data in 3D can be tricky due to the heterogeneous sizes.

For instance, point clouds can have different number of points, which means we can't always just concatenate the tensors on a batch axis.

Kaolin supports different batching strategies:

.. _exact:

Exact
~~~~~

Exact batching is the logical representation for homogeneous data.

For instance, if you sample the same numbers of points from a batch of meshes, you would just have a single tensor of shape :math:`(\text{batch_size}, \text{number_of_points}, 3)`.

.. _padded:

Padded
~~~~~~

Heterogeneous tensors are padded to identical dimensions with a constant value so that they can be concatenated on a batch axis. This is similar to padding for the batching of image data of different shapes.

.. note::
    The last dimension must always be of the size of the element, e.g. 3 for 3D points (element of point clouds) or 1 for a grayscale pixel (element of grayscale textures).

For instance, for two textures :math:`T_0` and :math:`T_1` of shape :math:`(32, 32, 3)` and :math:`(64, 16, 3)`
the batched tensor will be of shape :math:`(2, max(32, 64), max(32, 16), 3) = (2, 64, 32, 3)` and the padding value will be :math:`0`. :math:`T_0` will be padded on the 1st axis by :math:`32` while :math:`T_1` will be padded on the 2nd axis by :math:`16`.

You can also enforce a specific maximum shape (if you want to have a fix memory consumption or use optimization like cudnn algorithm selection).

For instance, you can force :math:`T_0` and :math:`T_1` to be batched with a maximum shape of :math:`(128, 128)`, the batched tensor will be of shape :math:`(2, 128, 128, 3)`, :math:`T_0` will be padded on the 1st axis and 2nd axis by 96 and :math:`T_1` will be padded on the 1st axis by :math:`64` and on the 2nd axis by :math:`112`.

For more information on how to do padded batching check :func:`kaolin.ops.batch.list_to_padded`

Related attributes:
...................
.. _padded_shape_per_tensor:

* :attr:`shape_per_tensor`: 2D :class:`torch.LongTensor` stores the shape of each sub-tensor except the last dimension in the padded tensor. E.g., in the example above :attr:`shape_per_tensor` would be ``torch.LongTensor([[32, 32], [64, 16]])``. Refer to :func:`kaolin.ops.batch.get_shape_per_tensor` for more information.

.. _packed:

Packed
~~~~~~

Heterogeneous tensors are reshaped to 2D :math:`(-1, \text{last_dimension})` and concatenated on the first axis. This is similar to packed sentences in NLP.

.. note::
    The last dimension must always be of the size of the element, e.g. 3 for 3D points (element of point clouds) or 1 for a grayscale pixel (element of grayscale textures).

For instance, for two textures :math:`T_0` and :math:`T_1` of shape :math:`(32, 32, 3)` and :math:`(64, 16, 3)`
The batched tensor will be of shape :math:`(32 * 32 + 64 * 16, 3)`. :math:`T_0` will be reshaped to :math:`(32 * 32, 3)` and :math:`T_1` will be reshaped :math:`(64 * 16, 3)`, before being concatenated on the first axis.

For more information on how to do padded batching check :func:`kaolin.ops.batch.list_to_packed`

Related attributes:
...................
.. _packed_shape_per_tensor:

* :attr:`shape_per_tensor`: 2D :class:`torch.LongTensor` stores the shape of each sub-tensor except the last dimension in the padded tensor. E.g., in the example above :attr:`shape_per_tensor` would be ``torch.LongTensor([[32, 32], [64, 16]])``. Refer to :func:`kaolin.ops.batch.get_shape_per_tensor` for more information.

.. _packed_first_idx:

* :attr:`first_idx`: 1D :class:`torch.LongTensor` stores the first index of each subtensor and the last index + 1 on the first axis in the packed tensor. E.g., in the example above :attr:`first_idx` would be ``torch.LongTensor([0, 1024, 2048])``. This attribute are used for delimiting each subtensor into the packed tensor, for instance, to slice or index. Refer to :func:`kaolin.ops.batch.get_first_idx` for more information.

API
---

.. automodule:: kaolin.ops.batch
   :platform: Windows-x86_64, Linux-x86_64
   :members:
   :undoc-members:
   :show-inheritance:
