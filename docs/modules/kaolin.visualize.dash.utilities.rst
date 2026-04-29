.. _kaolin.visualize.dash.utilities:

Other Utilities
===============

Supporting modules used by the viewer and app builder.

* :mod:`~kaolin.visualize.dash.assets_helper` -- locate and serve Kaolin's
  bundled Dash assets.
* :mod:`~kaolin.visualize.dash.behavior_manifest` -- read the build-time
  library behavior manifest shipped with the wheel.
* :mod:`~kaolin.visualize.dash.user_behavior_scan` -- discover behaviors
  registered in user-supplied asset directories at runtime.

The two discovery modules below are merged behind
:class:`~kaolin.visualize.dash.builtins.BehaviorLibrary`, so most code never
calls them directly. For the browser-side ``window.kaolin`` API, see the
:doc:`JavaScript API <kaolin.visualize.dash.javascript>`.

Asset serving
-------------

.. automodule:: kaolin.visualize.dash.assets_helper
   :members:
   :show-inheritance:

Behavior manifest
-----------------

.. automodule:: kaolin.visualize.dash.behavior_manifest
   :members:
   :show-inheritance:

User behavior scanning
----------------------

.. automodule:: kaolin.visualize.dash.user_behavior_scan
   :members:
   :show-inheritance:
