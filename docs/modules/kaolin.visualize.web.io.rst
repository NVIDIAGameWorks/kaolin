.. _kaolin.visualize.web.io:

kaolin.visualize.web.io
=======================

This module provides general binary encoding utilities, focusing
on simple types and encoding of PyTorch tensors for transferring
custom messages over the web, for example see . Sister encoding-decoding functions
are also available in javascript when using Kaolin's
`Plotly Dash <https://dash.plotly.com/>`_ integrations (See :doc:`kaolin.visualize.dash`).

Essential
---------

.. autofunction:: kaolin.visualize.web.io.encode_message
.. autofunction:: kaolin.visualize.web.io.to_binary
.. autofunction:: kaolin.visualize.web.io.from_binary
.. autoclass:: kaolin.visualize.web.io.BinaryIoDataType
   :members:
   :undoc-members:

Other API
---------

There should be no reason to use the lower-level helpers
below, but we expose them in case they are helpful. Encoding
particular types should be handled using the high-level
encoders :func:`to_binary` and :func:`from_binary`.

.. automodule:: kaolin.visualize.web.io
   :members:
   :exclude-members:
        to_binary,
        from_binary,
        encode_message,
        BinaryIoDataType
   :undoc-members:
   :show-inheritance:
