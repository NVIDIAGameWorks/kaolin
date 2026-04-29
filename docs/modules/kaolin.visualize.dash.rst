.. _kaolin.visualize.dash:

kaolin.visualize.dash
=====================

.. note::

   🌱 This toolkit is young and evolving. The high-level helpers are convenience
   shortcuts: when they do not fit your use case, drop down to plain
   `Plotly Dash <https://dash.plotly.com/>`_ / `tornado` and use the lower-level
   pieces directly.

Overview
--------

``kaolin.visualize.dash`` is a toolkit for building **interactive 3D/2D web
applications** with rich client-server communication on top of
`Plotly Dash <https://dash.plotly.com/>`_. It is designed for PyTorch workflows
where heavy work (rendering, inference, optimization) runs **server-side** while
the browser handles **interactive mouse / touch / keyboard input** and lightweight
client-side logic.

The centerpiece is :class:`~kaolin.visualize.dash.viewer.KaolinViewer`, a React
component exposed to Python. A viewer is composed of stacked, aligned **layers**
(canvas, SVG, ...) and configured with **behaviors** that map user interactions to
actions and to custom binary client-server messages over WebSockets. Around the
viewer, helpers auto-generate **UI controls** from typed Python sources and wire up
a complete Dash **application layout**.

How the pieces fit together:

* :doc:`Kaolin Viewer: Layers and Behaviors <kaolin.visualize.dash.viewer_group>`
  -- the :class:`~kaolin.visualize.dash.viewer.KaolinViewer` component, its
  :class:`~kaolin.visualize.dash.viewer.ViewerBuilder`, the catalog of built-in
  layers / behaviors, and the typed :class:`~kaolin.visualize.dash.option.OptionSpec`
  used to describe configurable options.
* :doc:`Building an App: Layout, Auto-UI & App Builder <kaolin.visualize.dash.app_group>`
  -- arrange a responsive page (:mod:`~kaolin.visualize.dash.layout`), optionally
  turn typed sources into Dash controls (:mod:`~kaolin.visualize.dash.auto_ui`),
  and assemble a runnable server (:mod:`~kaolin.visualize.dash.builder`).
* :doc:`JavaScript API <kaolin.visualize.dash.javascript>` -- the browser-side
  ``window.kaolin`` API: base classes for custom behaviors, viewer events, and
  binary message I/O for custom client-server communication.
* :doc:`Other utilities <kaolin.visualize.dash.utilities>` -- asset serving and
  behavior discovery helpers.

Example applications
--------------------

End-to-end sample apps that exercise this toolkit live under
``kaolin/app/`` in the source tree. They are the best starting point for seeing
the viewer, behaviors, and server-side rendering working together:

* ``kaolin/app/act_splat`` -- starter project: server-side Gaussian splat
  rendering with client-side splat control.
* ``kaolin/app/segment`` -- interactive **SAM2** point-prompt segmentation of
  Gaussian splats, combining server-side rendering with 2D selection.
* ``kaolin/app/splat_inpaint`` -- server-side 2D diffusion inpainting baked back
  into a 3D Gaussian splat scene via per-Gaussian color optimization.
* ``kaolin/app/mesh_edit`` -- server-side mesh rendering and editing
  (Axolotl3D) with 2D inpainting.
* ``kaolin/app/physics`` -- Simplicits-based interactive physics tools.

A minimal, self-contained demo combining a viewer with WebSocket communication is
in ``kaolin/app/dash_ws_combo_main.py``.

Detailed documentation
----------------------

.. toctree::
   :maxdepth: 2

   kaolin.visualize.dash.viewer_group
   kaolin.visualize.dash.app_group
   kaolin.visualize.dash.javascript
   kaolin.visualize.dash.utilities
