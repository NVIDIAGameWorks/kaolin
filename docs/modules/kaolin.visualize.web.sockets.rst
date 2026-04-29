.. _kaolin.visualize.web.sockets:

kaolin.visualize.web.sockets
============================

This module contains utlities for handling communication through
Web Sockets for client-server applications. The base communication
is handled by the :class:`kaolin.visualize.web.sockets.WebSocketHandlerManager`,
which can be configured with any number of available or
custom message handlers implementing
:class:`kaolin.visualize.web.sockets.SyncMessageHandlerProtocol`
or :class:`kaolin.visualize.web.sockets.AsyncMessageHandlerProtocol`.

Handler Protocols
-----------------

The library user only needs to understand and implement the
following protocols.

.. autoclass:: kaolin.visualize.web.sockets.MessageHandlerProtocol
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: kaolin.visualize.web.sockets.SyncMessageHandlerProtocol
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: kaolin.visualize.web.sockets.AsyncMessageHandlerProtocol
   :members:
   :undoc-members:
   :show-inheritance:

Available Handlers and Other API
--------------------------------
.. automodule:: kaolin.visualize.web.sockets
   :members:
   :exclude-members:
        MessageHandlerProtocol,
        SyncMessageHandlerProtocol,
        AsyncMessageHandlerProtocol,
        WebSocketHandlerManager
   :undoc-members:
   :show-inheritance:

WebSocket Handler
-----------------

The following websocket handler can be used with ``tornado``-based
applications for handling custom messages.

.. autoclass:: kaolin.visualize.web.sockets.WebSocketHandlerManager
   :members:
   :undoc-members:
   :show-inheritance:
