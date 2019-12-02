USD Tutorial 
============

Universal Scene Description (USD) is an open-source 3D scene 
description file format developed by Pixar developed to be 
versatile, extensible and interchangeable between different 3D 
tools.

USD supports asset referencing, making it suitable for
organizing entire datasets into interpretable, viewable 
subsets based on tags, class or other metadata label.


Creating a Mesh Dataset from a USD Scene
----------------------------------------

Kaolin supports importing meshes from a provided scene file. 
Let's use Pixar's `KitchenSet <http://graphics.pixar.com/usd/downloads.html>`_ for our example.

.. image:: images/kitchenset.png

Next we will create a mesh dataset from this scene. First, we need 
to set our environment variables:

.. code-block:: bash

    >>> source setenv.sh

This step is required for any interactions with USD (this will 
be addressed in future releases!). Now, we can create our dataset:

.. code-block:: python

    >>> from kaolin.datasets.usdfile import USDMeshes
    >>> usd_meshes = USDMeshes(usd_filepath='./data/Kitchen_set/Kitchen_set.usd')
    >>> len(usd_meshes)
    740

And just like that, we have a dataset of 740 diverse objects for our use!
Let's see what they look like.

    >>> from kaolin.visualize.vis_usd import VisUsd
    >>> vis = VisUsd()
    >>> vis.set_stage()
    >>> spacing = 200
    >>> for i, sample in enumerate(usd_meshes):
    >>>     attr, data = sample['attributes'], sample['data']
    >>>     max_x = int(math.sqrt(len(usd_meshes)))
    >>>     x = (i % max_x) * spacing
    >>>     y = (i // max_x) * spacing
    >>>     vis.visualize(mesh, object_path=f'/Root/Visualizer/{attr["name"]}', \
    >>>                   translation=(x, y, 0))


And opening the USD in your favourite USD viewer with a bit of styling, we get:

.. image:: images/kitchenset_meshes.png

Viewing USD Files
-----------------
USD files can be visualized using Pixar's USDView which you can obtain by visiting 
`https://developer.nvidia.com/usd <https://developer.nvidia.com/usd>`_ and selecting the 
corresponding platform under *.USD Pre-Built Libraries and Tools*. Note, USDView only supports
python 2.7.

Some Notes
----------

- Currently, \*.usd and \*.usda file extensions are supported. 
- When importing meshes, any mesh that has a varying number of vertices per face cannot be imported.