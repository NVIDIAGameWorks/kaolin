.. _kaolin.visualize.dash.javascript:

.. raw:: html

    <style>
        h1 {
            color: white;
            background-color: black;
            padding: 30px 10px;
        }

        .typedoc-if {
            width: 100%;
            height: 800px;
        }
    </style>

Kaolin's Javascript API
=======================

When using the :doc:`Kaolin viewer <kaolin.visualize.dash.viewer_group>` in a
Plotly Dash application, the javascript utilities below are exposed in the browser
under the ``window.kaolin`` namespace. Some functionality (such as behaviors) is
configured from the Python API, while other functions are designed to help you
implement client-side callbacks for custom Dash applications directly in
javascript.

* To **register your own behaviors** in plain javascript, see
  :ref:`Custom behaviors <dash-custom-behaviors>`.
* To **load that javascript** and otherwise extend an app, see
  :ref:`Customizing your app <dash-customizing>`.

.. note::
       The following API is in Javascript and will be available in the browser,
       under ``window.kaolin``.

Key client-side functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most useful entry points for client-side development, grouped by their
``window.kaolin`` namespace. Full signatures and the complete API are in the
embedded reference below.

``kaolin.io`` -- message encoding for client-server communication
-----------------------------------------------------------------

JSON is fine for small messages but inefficient for large tensor data such as
images. These helpers provide compact binary (and JSON) serialization of custom
messages; the sister functions on the python side live in
:mod:`~kaolin.visualize.web.io`. Messages carry a ``Map`` (javascript) / ``dict``
(python) of simple values or arrays; numeric arrays decode to typed arrays (e.g.
``Float32Array``) carrying an extra ``shape`` attribute, and image payloads decode
to ``Blob`` instances.

* ``kaolin.io.encodeMessage(tag, content, binary=true)`` -- encode a tagged
  message ready to send over the WebSocket / Dash bridge.
* ``kaolin.io.toBinary(value)`` / ``kaolin.io.fromBinary(buffer)`` -- (de)serialize
  an arbitrary value to/from a binary ``ArrayBuffer``.
* ``kaolin.io.toJSON(value)`` / ``kaolin.io.fromJSON(text)`` -- ``Map``-aware JSON
  variants, consistent with the binary forms.
* ``kaolin.io.downloadURL(filename, url)`` -- trigger a browser download.

``kaolin.core.event`` -- drive the viewer and behaviors from callbacks
----------------------------------------------------------------------

Use these from client-side Dash callbacks to control a running viewer by id
(``viewerId`` defaults to ``'kaolin-viewer'``):

* ``kaolin.core.event.requestBehaviorSetOption(behaviorId, name, value, viewerId?)``
  -- change a behavior option at runtime (the preferred, schema-validated path).
* ``kaolin.core.event.requestBehaviorActiveStatus(behaviorId, active, viewerId?)``
  -- enable or disable a behavior.
* ``kaolin.core.event.requestBehaviorReset(behaviorId, value?, viewerId?)`` --
  reset a behavior to a clean state.
* ``kaolin.core.event.requestMode(modeName, viewerId?)`` -- switch the viewer's
  active mode (modes are declared at build time via ``ViewerBuilder.add_mode``).

``kaolin.core.behavior`` -- build and register custom behaviors
---------------------------------------------------------------

* ``kaolin.core.behavior.BehaviorRegister.register(name, cls, description)`` --
  register a behavior class under the ``name`` you pass to ``add_behavior`` from
  Python.
* Base classes to extend: ``Behavior``, ``InteractiveBehavior``,
  ``CanvasBehavior``, ``CameraControllerBase``, ``MessageHandlerBase``.

See :ref:`Custom behaviors <dash-custom-behaviors>` for a complete, runnable
example.

Full reference
~~~~~~~~~~~~~~~

.. raw:: html

    <iframe class="typedoc-if"
            src="../typedoc/index.html"
            title="Javascript documentation">
    </iframe>
