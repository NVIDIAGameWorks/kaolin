.. _kaolin.visualize:

kaolin.visualize
================

This module contains components for interactive visualization.

Interactive Web Client-Server Applications
------------------------------------------

The most recent
addition to this module is a full-featured toolkit to build **interactive Web
prototypes** with complex client-server communication, for example running
inference or even rendering on the server-side, while supporting
complex mouse or touch interactions in the brower. This functionality is built
over `Plotly Dash <https://dash.plotly.com/>`_ for python-built
Web applications, and has extensive support for interactive
3D applications and custom client-server workflows for PyTorch.

Dive in here:

* :doc:`kaolin.visualize.dash` - 👈 start here; high-level API, examples and tutorials
* :doc:`kaolin.visualize.web` - general Web utilities

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   kaolin.visualize.dash
   kaolin.visualize.web


IPython / Jupyter Notebook Visualization
----------------------------------------

For interactive visualization in IPython / Jupyter notebooks,
see the following documentation.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   kaolin.visualize.ipython

Other
-----

.. automodule:: kaolin.visualize
   :members:
   :inherited-members:
   :exclude-members:
       dash,
       web,
       ipython,
       BaseIpyVisualizer,
       IpyFirstPersonVisualizer,
       IpyTurntableVisualizer
   :undoc-members:
   :show-inheritance:


