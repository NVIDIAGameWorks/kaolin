.. _kaolin.io.usd:

kaolin.io.usd
=============

Universal Scene Description
---------------------------

Universal Scene Description (USD) is an open-source 3D scene description file format developed by Pixar and designed to be versatile, extensible and interchangeable between different 3D tools.

Single models and animations as well as large organized scenes composed of any number of assets can be defined in USD, making it suitable for organizing entire datasets into interpretable,
subsets based on tags, class or other metadata label.

Kaolin includes base I/O operations for USD and also leverages this format to export 3D checkpoints. Use kaolin.io.usd to read and write USD files (try :code:`tutorials/usd_kitcheset.py`),
and :code:`kaolin.visualize.Timelapse` to export 3D checkpoints (try :code:`tutorials/visualize_main.py`).

As a first step to familiarizing yourself with USD, we suggest following this `tutorial <https://developer.nvidia.com/usd>`_.
More tutorials and documentation can be found `here <https://graphics.pixar.com/usd/docs/Introduction-to-USD.html>`_.


Viewing USD Files
~~~~~~~~~~~~~~~~~
USD files can be visualized with realtime pathtracing using the [Omniverse Kaolin App](https://docs.omniverse.nvidia.com/app_kaolin/app_kaolin/user_manual.html#training-visualizer).
Alternatively, you may use Pixar's USDView which can be obtained by visiting
`https://developer.nvidia.com/usd <https://developer.nvidia.com/usd>`_ and selecting the
corresponding platform under *USD Pre-Built Libraries and Tools*.


API
---

Functions
---------

.. automodule:: kaolin.io.usd
    :members:
    :exclude-members:
        mesh_return_type
