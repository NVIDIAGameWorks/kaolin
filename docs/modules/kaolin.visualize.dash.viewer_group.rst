.. _kaolin.visualize.dash.viewer_group:

Kaolin Viewer: Layers and Behaviors
===================================

This page covers the interactive viewer itself, the catalog of built-in behaviors
it understands, how to write your own behaviors, and the typed option schemas that
tie Python and the browser together.

Overview & design
-----------------

The :class:`~kaolin.visualize.dash.viewer.KaolinViewer` is a React component,
exposed to Python, that provides an interactive 2D/3D surface inside a
`Plotly Dash <https://dash.plotly.com/>`_ app. Its design has two core concepts:

* **Layers** -- the viewer is composed of stacked, pixel-aligned layers
  (``canvas``, ``svg``, ...). Layers share the same coordinate space and are
  composited with opacity, so you can, for example, draw a server-rendered image
  on one canvas and let the user annotate on another above it.
* **Behaviors** -- named units of client-side logic that map user interaction
  (pointer / touch / keyboard) and incoming messages to actions. A behavior may
  paint on a layer, drive the camera, or stream data to/from the server.

Everything -- layers and behaviors alike -- **runs client-side** in the browser.
Behaviors are what make rich **client-server** workflows possible: a behavior can
serialize the camera or a custom binary message and send it over a WebSocket, and
another behavior can receive server frames and draw them. Heavy work (rendering,
inference) stays on the server while interaction stays fluid in the browser. The
browser-side helpers used to build behaviors live in the
:doc:`JavaScript API <kaolin.visualize.dash.javascript>`.

.. important::

   **Never construct** :class:`~kaolin.visualize.dash.viewer.KaolinViewer`
   **directly.** Always assemble a viewer with
   :class:`~kaolin.visualize.dash.viewer.ViewerBuilder` and call
   :meth:`~kaolin.visualize.dash.viewer.ViewerBuilder.build`. The builder
   allocates globally-unique layer/behavior ids, validates behavior options
   against their schemas, and wires the clientside callbacks that a hand-built
   component would be missing.

Getting started
---------------

A typical flow is: create a :class:`~kaolin.visualize.dash.viewer.ViewerBuilder`,
add one or more layers, attach behaviors by name, optionally generate UI controls
for those behaviors, then :meth:`~kaolin.visualize.dash.viewer.ViewerBuilder.build`
the Dash component:

.. code-block:: python

    from kaolin.visualize.dash import ViewerBuilder
    from kaolin.render.easy_render import default_camera

    builder = ViewerBuilder(camera=default_camera(500))

    # 1. Stack a layer (canvas/svg) inside the viewer.
    draw_layer = builder.add_layer('canvas')

    # 2. Attach a behavior by its registered name (see Built-in Behaviors).
    draw_id = builder.add_behavior('drawing', active_layer_id=draw_layer,
                                   options={'color': '#ffcc00'})

    # 3. Auto-generate UI controls for that behavior's options. Returns Dash
    #    components you insert into your app layout.
    option_controls = builder.add_user_behavior_options(draw_id)

    # 4. Build the Dash component. NEVER instantiate KaolinViewer yourself.
    viewer = builder.build()

From here, dive into the detailed sections:

* :ref:`dash-configuring-viewer` -- layers, cameras, remote rendering, and the
  full :class:`~kaolin.visualize.dash.viewer.ViewerBuilder` API.
* :ref:`dash-builtin-behaviors` -- discover ready-made behaviors with
  :class:`~kaolin.visualize.dash.builtins.BehaviorLibrary` and expose their
  options as UI.
* :ref:`dash-custom-behaviors` -- register your own behavior in plain JavaScript.
* :ref:`dash-behavior-schemas` -- how
  :class:`~kaolin.visualize.dash.option.OptionSpec` schemas drive validation and
  auto-UI.

.. _dash-configuring-viewer:

Configuring the viewer
----------------------

:class:`~kaolin.visualize.dash.viewer.ViewerBuilder` is the single entry point for
assembling a viewer. The most common methods are:

* :meth:`~kaolin.visualize.dash.viewer.ViewerBuilder.add_layer` -- add a
  ``canvas`` / ``svg`` layer; returns a builder-level layer id.
* :meth:`~kaolin.visualize.dash.viewer.ViewerBuilder.add_behavior` -- attach a
  behavior by registered name, optionally bound to a layer and configured with
  ``options``.
* :meth:`~kaolin.visualize.dash.viewer.ViewerBuilder.add_user_behavior_options`
  -- auto-generate UI controls for a behavior's options (see
  :ref:`dash-builtin-behaviors`).
* :meth:`~kaolin.visualize.dash.viewer.ViewerBuilder.add_remote_rendering` --
  one-call setup for server-side rendering: streams the camera to the server and
  draws the frames it pushes back.
* :meth:`~kaolin.visualize.dash.viewer.ViewerBuilder.build` -- produce the
  :class:`~kaolin.visualize.dash.viewer.KaolinViewer` Dash component.

Server-side rendering in a few lines:

.. code-block:: python

    builder = ViewerBuilder(camera=default_camera(800))
    render_layer = builder.add_layer('canvas')
    # Stream the camera to the server and draw the frames it returns.
    builder.add_remote_rendering(active_layer_id=render_layer,
                                 connection_id='main-ws')
    viewer = builder.build()

.. automodule:: kaolin.visualize.dash.viewer
   :members:
   :show-inheritance:

.. _dash-builtin-behaviors:

Built-in behaviors
------------------

Kaolin ships a library of ready-made behaviors (drawing, SVG annotation, camera
sending, remote-image drawing, ...). Use them by passing their name to
:meth:`~kaolin.visualize.dash.viewer.ViewerBuilder.add_behavior`.
:class:`~kaolin.visualize.dash.builtins.BehaviorLibrary` is the single Python
entry point to discover what is available -- it merges the build-time library
manifest with any behaviors found in your own asset directories:

.. code-block:: python

    from kaolin.visualize.dash import BehaviorLibrary

    print(BehaviorLibrary)                  # human-readable summary
    names = BehaviorLibrary.names()         # available behavior names
    meta = BehaviorLibrary.meta('drawing')  # BehaviorMeta (incl. option schema)

A behavior's option schema is also what powers **auto-generated UI**. Call
:meth:`~kaolin.visualize.dash.viewer.ViewerBuilder.add_user_behavior_options`
with a behavior id to get Dash controls (plus a hidden store for per-tab
persistence); each control edit is pushed to the live behavior in the browser
without a server roundtrip:

.. code-block:: python

    draw_id = builder.add_behavior('drawing', active_layer_id=draw_layer)

    # All uiBound options, or a subset by name / overriding OptionSpec.
    controls = builder.add_user_behavior_options(draw_id)
    controls = builder.add_user_behavior_options(draw_id, options=['thickness'])

    # ...insert `controls` into your Dash layout (e.g. a sidebar).

.. automodule:: kaolin.visualize.dash.builtins
   :members:
   :show-inheritance:

.. _dash-custom-behaviors:

Custom behaviors
----------------

Behaviors are resolved by name in the browser, so a custom behavior is just a
class registered under a name -- **plain JavaScript is enough, no TypeScript or
build step required**. Drop a script (e.g. ``custom.js``) into your app's served
assets; the ``kaolin`` namespace is loaded automatically (see the
:doc:`JavaScript API <kaolin.visualize.dash.javascript>`).

.. code-block:: javascript

    // custom.js -- loaded in the browser; `kaolin` is available globally.
    const { CanvasBehavior, BehaviorRegister, OptionKind } = kaolin.core.behavior;

    class DotBehavior extends CanvasBehavior {
        // Schema drives Python-side validation and auto-UI (see Behavior Schemas).
        static schema = { radius: { kind: OptionKind.INT, default: 6, min: 1, max: 40 } };

        // Re-render whenever an option is edited from Python.
        updateForOptions() { this.redraw(); }

        // CanvasBehavior provides `this.element` / `this.ctx` and pointer handlers.
        onPointerDown(event) { this.x = event.offsetX; this.y = event.offsetY; this.redraw(); }

        redraw() {
            if (!this.ctx || this.x === undefined) return;
            this.ctx.beginPath();
            this.ctx.arc(this.x, this.y, this.options.radius, 0, 2 * Math.PI);
            this.ctx.fill();
        }
    }

    // 'dot' is the name you pass to add_behavior(...) from Python.
    BehaviorRegister.register('dot', DotBehavior, 'Draws a dot where you click.');

Then use it from Python exactly like a built-in:

.. code-block:: python

    dot_layer = builder.add_layer('canvas')
    dot_id = builder.add_behavior('dot', active_layer_id=dot_layer,
                                  options={'radius': 10})

See the :doc:`JavaScript API <kaolin.visualize.dash.javascript>` for the base
classes (``Behavior``, ``InteractiveBehavior``, ``CanvasBehavior``,
``CameraControllerBase``, ``MessageHandlerBase``), the event interfaces, and the
binary message I/O utilities used for custom client-server communication.

.. _dash-behavior-schemas:

Behavior schemas
----------------

Each behavior declares an **option schema**: the set of configurable parameters,
their types, bounds, and defaults. On the browser side this is a `Zod
<https://zod.dev/>`_ schema (or, in plain JS, the ``static schema`` object shown
above); at build time it is projected to JSON and read back on the Python side as
:class:`~kaolin.visualize.dash.option.OptionSpec` objects.

A single :class:`~kaolin.visualize.dash.option.OptionSpec` describes one option
and is the unit that flows through the whole system:

* it round-trips through JSON (Python ⇄ TypeScript) via ``as_dict`` / ``from_dict``;
* :class:`~kaolin.visualize.dash.builtins.BehaviorLibrary` exposes the specs of
  every known behavior;
* :meth:`~kaolin.visualize.dash.viewer.ViewerBuilder.add_user_behavior_options`
  turns specs into Dash controls and validates option values against them.

You rarely construct one by hand, but you can -- for example to override a bound
or define an option for a custom behavior that has no manifest:

.. code-block:: python

    from kaolin.visualize.dash.option import OptionSpec, OptionKind

    thickness = OptionSpec(
        name='thickness', kind=OptionKind.INT, default=4, minimum=1, maximum=20)

    # Expose just this option, with our overridden bounds, as UI for a behavior.
    controls = builder.add_user_behavior_options(draw_id, options=[thickness])

.. automodule:: kaolin.visualize.dash.option
   :members:
   :show-inheritance:
