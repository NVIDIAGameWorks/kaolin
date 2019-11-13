Datasets
=================================

One of the very first things you would want to do, for a new 3D deep learning 
application, is to load data into a format that PyTorch can operate upon. 
With Kaolin, the process is a breeze, as we have done most of the hard work 
for you!

.. contents::
    :local:

ShapeNet
--------

.. image:: /_static/img/ShapeNet.png

`ShapeNet <https://shapenet.org/>`_ is an extremely popular (and huge!) repository 
of objects encoded as triangle meshes, and is often used in 3D deep learning. 
However, you will first need to obtain access to the dataset (agree to terms of use, 
download, unzip), and then you can leverage the power of Kaolin to load up ShapeNet 
data in several formats.

Kaolin supports multiple formats in which you can load a ShapeNet class (eg. meshes, 
pointclouds, voxels, signed distance functions, and more). Although ShapeNet objects 
are usually available only as meshes, Kaolin internally converts to other 
representations and returns them. This means that, the first time you load a 
ShapeNet category in a non-native (i.e., non mesh) format, it will probably take a 
while, as it is internally converting, and then caching data.

Assuming that the ShapeNet directory is located at a path `shapenet_dir`, here's 
how to load meshes from the ShapeNet `chair` category.

.. code-block:: python

    >>> from kaolin.datasets import shapenet
    >>> from torch.utils.data import DataLoader
    >>> meshes = shapenet.ShapeNet_Meshes(root=shapenet_dir, categories=['plane'])

.. image:: /_static/img/planes_mesh.png


Loading multiple categories is straightforward too.

.. code-block:: python
    
    >>> meshes = shapenet.ShapeNet_Meshes(root=shapenet_dir, 
    categories=['chair', 'bowl'])

You can either specify the `categories` by their plaintext names 
(eg. ``chair``, ``bowl``, etc.) or use their ``synset_id`` 
(eg. ``03636649``, ``02924116``, etc.)

Suppose you need a different representation than meshes. You can easily load different 
representations, such as voxels and point clouds.

.. code-block:: python
    
    >>> voxels = shapenet.ShapeNet_Voxels(root=shapenet_dir, categories=['plane'])
    >>> points = shapenet.ShapeNet_Points(root=shapenet_dir, categories=['plane'])

This can now be used to initialize a Pytorch dataloader, in a similar way as 
you would for 2D data.

.. code-block:: python

    >>> voxel_dataloader = DataLoader(voxels, batch_size=10, shuffle=True, num_workers=8)
    >>> point_dataloader = DataLoader(points, batch_size=10, shuffle=True, num_workers=8)


.. image:: /_static/img/planes_voxels.png
    :width: 49 %
.. image:: /_static/img/planes_pc.png
    :width: 49 %


ModelNet
--------

ModelNet voxels can be loaded in a similar way:

.. code-block:: python

    >>> from kaolin.datasets import modelnet
    >>> voxels = modelnet.ModelNet(root=shapenet_dir, categories=['plane'])
    >>> dataloader = DataLoader(voxels, batch_size=10, shuffle=True, num_workers=8)


SHREC16
-------
.. image:: /_static/img/shrec.png

SHREC is a dataset that was curated for the 
`Large-Scale 3D Retrieval From ShapeNet Core55` challenge at Eurographics 2016. 
We implement a mesh dataloader for SHREC16.

.. code-block:: python

    >>> from kaolin.datasets import shrec
    >>> meshes = shrec.SHREC16(root=shapenet_dir, categories=['plane'])
    >>> dataloader = DataLoader(meshes, batch_size=10, shuffle=True, num_workers=8)


More to come
---------------
Kaolin supports a bunch of other datasets too. Stay tuned to this space for more tutorials.
