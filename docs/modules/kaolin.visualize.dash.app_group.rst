.. _kaolin.visualize.dash.app_group:

Building an App: Layout, Auto-UI & App Builder
==============================================

These modules help you assemble a complete, runnable Dash application: arrange a
responsive page, optionally auto-generate controls from typed Python, wire up
WebSocket communication, and assemble the server. They are **convenience helpers**
-- you can mix in plain Dash components, callbacks, and layouts wherever you like.

Tutorial
--------

The snippet below shows a typical end-to-end flow; each step links to its detailed
section. Only the layout and the app builder are essential -- the viewer and the
auto-generated UI are optional and shown here to illustrate a full app.

.. code-block:: python

    from dataclasses import dataclass

    from kaolin.visualize.dash import ViewerBuilder, auto_ui
    from kaolin.visualize.dash.layout import StandardLayoutHelper
    from kaolin.visualize.dash.builder import WebappBuilder
    from kaolin.render.easy_render import default_camera

    # 1. (Optional) Build an interactive viewer (server-side rendering over a
    #    WebSocket). Skip this for a non-3D app.
    vb = ViewerBuilder(camera=default_camera(800))
    layer = vb.add_layer('canvas')
    vb.add_remote_rendering(active_layer_id=layer, connection_id='main-ws')
    viewer = vb.build()

    # 2. (Optional) Auto-generate UI controls from a typed dataclass. You can
    #    instead hand-write any plain Dash components and callbacks.
    @dataclass
    class RenderSettings:
        exposure: float = 1.0
        wireframe: bool = False

    controls, field_specs, store = auto_ui.controls_from_dataclass(
        RenderSettings(), id_prefix='render', make_store=True)

    # 3. Arrange your content in a responsive page (plain Dash components welcome).
    layout = StandardLayoutHelper(title='My Kaolin App')
    layout.add_sidebar()
    layout.add_sidebar_components(controls + [store])
    layout.add_main_content([viewer])

    # 4. Assemble the server, optionally registering a WebSocket handler.
    builder = WebappBuilder(debug=True)
    builder.set_layout_helper(layout)
    app = builder.build(ws_handlers=[(EchoHandler, ())])
    # app.run_server(...)

The pieces, in order:

#. **(Optional) Add a viewer** built with a
   :class:`~kaolin.visualize.dash.viewer.ViewerBuilder` -- see the dedicated
   :doc:`Kaolin Viewer: Layers and Behaviors <kaolin.visualize.dash.viewer_group>`
   page.
#. **(Optional) Auto-generate UI** from a dataclass (or function, or behavior)
   with :mod:`~kaolin.visualize.dash.auto_ui` -- or just use plain Dash components
   and callbacks. See :ref:`dash-auto-ui`.
#. **Lay it out** with a :class:`~kaolin.visualize.dash.layout.StandardLayoutHelper`
   -- see :ref:`dash-layout`.
#. **Assemble the server** with
   :class:`~kaolin.visualize.dash.builder.WebappBuilder`, optionally registering a
   WebSocket handler for custom client-server communication -- see
   :ref:`dash-app-builder` and :ref:`dash-websockets`.

See :ref:`dash-customizing` for going beyond these helpers.

.. _dash-auto-ui:

Auto-UI
-------

Auto-UI is **entirely optional**: it is just a shortcut for generating standard
Dash controls. You are free to hand-write any `Dash <https://dash.plotly.com/>`_
components (``dcc.Slider``, ``dbc.Input``, ...) and wire them with ordinary Dash
``@callback`` / clientside callbacks instead -- mix and match as you like.

When it helps, :mod:`~kaolin.visualize.dash.auto_ui` builds Dash controls
automatically from typed sources -- a dataclass instance
(:func:`~kaolin.visualize.dash.auto_ui.controls_from_dataclass`), a callable's
signature (:func:`~kaolin.visualize.dash.auto_ui.controls_from_function`), or a
list of :class:`~kaolin.visualize.dash.option.OptionSpec` objects
(:func:`~kaolin.visualize.dash.auto_ui.make_controls`). Generated controls get
deterministic ids of the form ``f"{id_prefix}-{field_name}"`` so they are easy to
wire into callbacks, and an optional backing :class:`dash.dcc.Store` can be seeded
from the instance.

.. code-block:: python

    from dataclasses import dataclass
    from kaolin.visualize.dash import auto_ui

    @dataclass
    class RenderSettings:
        exposure: float = 1.0
        wireframe: bool = False

    controls, field_specs, store = auto_ui.controls_from_dataclass(
        RenderSettings(), id_prefix='render', make_store=True)

.. note::

   To generate controls for a **behavior's** options rather than a dataclass,
   use :meth:`~kaolin.visualize.dash.viewer.ViewerBuilder.add_user_behavior_options`,
   which is described under
   :ref:`Built-in behaviors <dash-builtin-behaviors>` on the viewer page. It
   builds on the same :mod:`~kaolin.visualize.dash.auto_ui` machinery, additionally
   pushing each edit straight to the live behavior in the browser.

.. automodule:: kaolin.visualize.dash.auto_ui
   :members:
   :show-inheritance:

.. _dash-layout:

Layout
------

:mod:`~kaolin.visualize.dash.layout` provides the
:class:`~kaolin.visualize.dash.layout.AppLayoutHelper` protocol and
:class:`~kaolin.visualize.dash.layout.StandardLayoutHelper`, a responsive
(Bootstrap) scaffold for sidebar + viewport apps. Build a layout by adding a
sidebar, dropping components/sections into it, and placing your viewer(s) in the
main content area:

.. code-block:: python

    from kaolin.visualize.dash.layout import StandardLayoutHelper

    layout = StandardLayoutHelper(title='My Kaolin App')
    layout.add_sidebar()
    layout.add_sidebar_components(controls + [store])
    layout.add_main_content([viewer])
    # or: layout.add_viewer_grid([viewer_a, viewer_b], columns=2)

.. automodule:: kaolin.visualize.dash.layout
   :members:
   :show-inheritance:

.. _dash-websockets:

WebSocket handlers
------------------

Rich client-server communication (server-side rendering, streaming geometry,
custom binary messages) flows over WebSockets. On the server you implement a
**message handler** -- a class satisfying
:class:`~kaolin.visualize.web.sockets.SyncMessageHandlerProtocol` (or its async
sibling :class:`~kaolin.visualize.web.sockets.AsyncMessageHandlerProtocol`) -- and
register it with the app builder. A handler declares which message *tags* it
accepts and reacts to them, writing messages back to the client:

.. code-block:: python

    from kaolin.visualize.web.sockets import SyncMessageHandlerProtocol

    class EchoHandler(SyncMessageHandlerProtocol):
        def accepted_message_tags(self):
            return ['ping']

        def on_connection_open(self, write_message_fn):
            write_message_fn({'status': 'connected'}, False)  # False -> JSON

        def on_message(self, message_tag, message_content, write_message_fn):
            if message_tag == 'ping':
                write_message_fn({'pong': message_content}, False)

Handlers are registered by passing ``ws_handlers`` to
:meth:`~kaolin.visualize.dash.builder.WebappBuilder.build` as a list of
``(handler_class, args)`` (or ``(handler_class, args, kwargs)``) tuples; a fresh
handler instance is constructed **per connection**, so per-client state is never
shared:

.. code-block:: python

    app = builder.build(ws_handlers=[(EchoHandler, ())])

For large payloads such as images or tensors, use binary messages: encode them on
the server with :mod:`kaolin.visualize.web.io` and decode them in the browser with
the matching utilities in the
:doc:`JavaScript API <kaolin.visualize.dash.javascript>`. The clientside
counterpart that opens these connections and dispatches messages by tag is
``core/sockets.ts``, also documented there.

The full handler API -- the protocols, ready-made handlers (e.g.
:class:`~kaolin.visualize.web.sockets.AnyRendererMessageHandler`), and the
:class:`~kaolin.visualize.web.sockets.WebSocketHandlerManager` -- lives on the
dedicated :doc:`kaolin.visualize.web.sockets <kaolin.visualize.web.sockets>` page.

.. _dash-customizing:

Customizing your app
--------------------

:class:`~kaolin.visualize.dash.builder.WebappBuilder` is deliberately thin; reach
for plain Dash whenever it serves you better, and use the builder hooks below to
extend an app:

* **Custom client-side behaviors** -- drop a plain ``.js`` file defining and
  registering your behaviors (see
  :ref:`Custom behaviors <dash-custom-behaviors>` on the viewer page) into your
  app's ``assets/`` directory. Dash serves ``assets/`` automatically and loads
  ``.js`` files on page load, so the ``kaolin`` namespace and your
  ``BehaviorRegister.register(...)`` calls are available with no extra wiring.
  To make Python aware of those behaviors (e.g. for
  :meth:`~kaolin.visualize.dash.viewer.ViewerBuilder.add_user_behavior_options`),
  also call
  :meth:`BehaviorLibrary.register_user_directory(<app>/assets) <kaolin.visualize.dash.builtins.BehaviorLibrary.register_user_directory>`.
* **Extra scripts / stylesheets** --
  :meth:`~kaolin.visualize.dash.builder.WebappBuilder.add_extra_scripts` and
  :meth:`~kaolin.visualize.dash.builder.WebappBuilder.add_extra_stylesheets`.
* **ES module imports** (e.g. third-party libraries) --
  :meth:`~kaolin.visualize.dash.builder.WebappBuilder.add_importmap_item`.
* **Serving your own files / directories** --
  :meth:`~kaolin.visualize.dash.builder.WebappBuilder.add_static_file` and
  :meth:`~kaolin.visualize.dash.builder.WebappBuilder.add_static_dir`.
* **Raw HTML in the document body** --
  :meth:`~kaolin.visualize.dash.builder.WebappBuilder.add_raw_body_html`.
* **Persisted, per-tab user settings** --
  :meth:`~kaolin.visualize.dash.builder.WebappBuilder.add_user_settings`.

For anything beyond these, build the :class:`dash.Dash` app (and tornado server)
yourself -- nothing here is required.

.. _dash-app-builder:

App builder
-----------

:class:`~kaolin.visualize.dash.builder.WebappBuilder` ties layout, assets, user
settings, and WebSocket handlers into a single Dash + tornado server.
:class:`~kaolin.visualize.dash.builder.SessionRegistry` bridges Dash callbacks
and per-tab WebSocket state.

.. automodule:: kaolin.visualize.dash.builder
   :members:
   :show-inheritance:
